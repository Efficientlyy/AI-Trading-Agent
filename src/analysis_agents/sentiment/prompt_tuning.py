"""Prompt tuning system for LLM sentiment analysis.

This module provides a system for automatically tuning and optimizing
prompts for large language models based on performance feedback and
evaluation metrics.
"""

import asyncio
import json
import os
import random
import time
import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Set, Tuple, Union

import numpy as np

from src.common.config import config
from src.common.logging import get_logger
from src.common.events import event_bus, Event
from src.common.caching import Cache
from src.analysis_agents.sentiment.performance_tracker import performance_tracker


class PromptVariationType(Enum):
    """Types of prompt variations that can be generated."""
    INSTRUCTION_ADJUSTMENT = "instruction_adjustment"
    FORMAT_ADJUSTMENT = "format_adjustment"
    SYSTEM_ROLE_ADJUSTMENT = "system_role_adjustment"
    EXAMPLE_ADDITION = "example_addition"
    CONTEXT_ENRICHMENT = "context_enrichment"
    TEMPERATURE_ADJUSTMENT = "temperature_adjustment"


class PromptVersion:
    """Represents a version of a prompt template."""
    
    def __init__(
        self,
        prompt_id: str,
        prompt_type: str,
        template: str,
        version: int,
        parent_id: Optional[str] = None,
        variation_type: Optional[PromptVariationType] = None,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Initialize a prompt version.
        
        Args:
            prompt_id: Unique ID for this prompt
            prompt_type: Type of the prompt (e.g., "sentiment_analysis")
            template: The prompt template string
            version: Version number
            parent_id: ID of the parent prompt (if this is a variation)
            variation_type: Type of variation from parent
            description: Description of this prompt version
            metadata: Additional metadata
        """
        self.prompt_id = prompt_id
        self.prompt_type = prompt_type
        self.template = template
        self.version = version
        self.parent_id = parent_id
        self.variation_type = variation_type
        self.description = description or f"Version {version} of {prompt_type} prompt"
        self.metadata = metadata or {}
        self.created_at = datetime.utcnow().isoformat()
        
        # Performance metrics
        self.usage_count = 0
        self.success_count = 0
        self.error_count = 0
        self.total_latency = 0
        self.accuracy = 0.0
        self.calibration_error = 0.0
        self.last_used = None
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert prompt version to dictionary.
        
        Returns:
            Dictionary representation
        """
        return {
            "prompt_id": self.prompt_id,
            "prompt_type": self.prompt_type,
            "template": self.template,
            "version": self.version,
            "parent_id": self.parent_id,
            "variation_type": self.variation_type.value if self.variation_type else None,
            "description": self.description,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "usage_count": self.usage_count,
            "success_count": self.success_count,
            "error_count": self.error_count,
            "total_latency": self.total_latency,
            "accuracy": self.accuracy,
            "calibration_error": self.calibration_error,
            "last_used": self.last_used
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PromptVersion':
        """Create a prompt version from dictionary.
        
        Args:
            data: Dictionary representation
            
        Returns:
            PromptVersion instance
        """
        prompt = cls(
            prompt_id=data["prompt_id"],
            prompt_type=data["prompt_type"],
            template=data["template"],
            version=data["version"],
            parent_id=data.get("parent_id"),
            variation_type=PromptVariationType(data["variation_type"]) if data.get("variation_type") else None,
            description=data.get("description"),
            metadata=data.get("metadata", {})
        )
        
        # Set performance metrics
        prompt.usage_count = data.get("usage_count", 0)
        prompt.success_count = data.get("success_count", 0)
        prompt.error_count = data.get("error_count", 0)
        prompt.total_latency = data.get("total_latency", 0)
        prompt.accuracy = data.get("accuracy", 0.0)
        prompt.calibration_error = data.get("calibration_error", 0.0)
        prompt.last_used = data.get("last_used")
        prompt.created_at = data.get("created_at", prompt.created_at)
        
        return prompt
    
    def record_usage(self, success: bool, latency_ms: float) -> None:
        """Record usage statistics for this prompt.
        
        Args:
            success: Whether the call was successful
            latency_ms: Latency in milliseconds
        """
        self.usage_count += 1
        self.last_used = datetime.utcnow().isoformat()
        
        if success:
            self.success_count += 1
            self.total_latency += latency_ms
        else:
            self.error_count += 1
    
    def update_metrics(self, accuracy: float, calibration_error: float) -> None:
        """Update performance metrics for this prompt.
        
        Args:
            accuracy: Accuracy metric
            calibration_error: Calibration error metric
        """
        # Weighted update to avoid wild swings from small sample sizes
        if self.usage_count == 0:
            self.accuracy = accuracy
            self.calibration_error = calibration_error
        else:
            # Weight existing metrics more heavily for stability
            weight = min(0.8, self.usage_count / (self.usage_count + 10))
            self.accuracy = (weight * self.accuracy) + ((1 - weight) * accuracy)
            self.calibration_error = (weight * self.calibration_error) + ((1 - weight) * calibration_error)
    
    def get_average_latency(self) -> float:
        """Get the average latency for successful calls.
        
        Returns:
            Average latency in milliseconds
        """
        if self.success_count == 0:
            return 0
        return self.total_latency / self.success_count
    
    def get_success_rate(self) -> float:
        """Get the success rate for this prompt.
        
        Returns:
            Success rate (0-1)
        """
        if self.usage_count == 0:
            return 1.0
        return self.success_count / self.usage_count
    
    def get_performance_score(self) -> float:
        """Calculate overall performance score.
        
        Returns:
            Performance score (0-1)
        """
        # Combine multiple metrics into a single score
        success_weight = 0.2
        accuracy_weight = 0.5
        latency_weight = 0.1
        calibration_weight = 0.2
        
        success_rate = self.get_success_rate()
        avg_latency = self.get_average_latency()
        
        # Normalize latency to 0-1 range (lower is better)
        # Assume 3000ms is a reasonable upper bound
        norm_latency = max(0, min(1, 1 - (avg_latency / 3000)))
        
        # Normalize calibration error (lower is better)
        norm_calibration = max(0, min(1, 1 - (self.calibration_error * 2)))
        
        # Calculate weighted score
        score = (
            success_rate * success_weight +
            self.accuracy * accuracy_weight +
            norm_latency * latency_weight +
            norm_calibration * calibration_weight
        )
        
        return score


class PromptTuningSystem:
    """System for tuning and optimizing LLM prompts.
    
    This system tracks prompt performance, generates variations,
    and automatically selects the best performing prompts.
    """
    
    def __init__(self):
        """Initialize the prompt tuning system."""
        self.logger = get_logger("analysis_agents", "prompt_tuning")
        
        # Config
        self.tuning_enabled = config.get("sentiment_analysis.prompt_tuning.enabled", False)
        self.auto_optimize = config.get("sentiment_analysis.prompt_tuning.auto_optimize", False)
        self.experiment_ratio = config.get("sentiment_analysis.prompt_tuning.experiment_ratio", 0.2)
        self.min_usage_optimize = config.get("sentiment_analysis.prompt_tuning.min_usage_optimize", 20)
        self.max_versions = config.get("sentiment_analysis.prompt_tuning.max_versions", 5)
        self.evaluation_interval = config.get("sentiment_analysis.prompt_tuning.evaluation_interval", 3600)
        self.storage_path = config.get("sentiment_analysis.prompt_tuning.storage_path", "data/prompts")
        
        # State
        self.prompt_versions: Dict[str, List[PromptVersion]] = {}
        self.active_prompts: Dict[str, PromptVersion] = {}
        self.prompts_in_use: Dict[str, Dict[str, Any]] = {}  # Track ongoing requests
        self.last_evaluation_time = 0
        self.experiment_ids: Set[str] = set()  # Ongoing experiments
        
        # Create directory for prompt storage
        os.makedirs(self.storage_path, exist_ok=True)
        
        # Setup template generators
        self._initialize_template_generators()
    
    def _initialize_template_generators(self) -> None:
        """Initialize template generators for prompt variations."""
        self.variation_generators = {
            PromptVariationType.INSTRUCTION_ADJUSTMENT: self._generate_instruction_variation,
            PromptVariationType.FORMAT_ADJUSTMENT: self._generate_format_variation,
            PromptVariationType.SYSTEM_ROLE_ADJUSTMENT: self._generate_system_role_variation,
            PromptVariationType.EXAMPLE_ADDITION: self._generate_example_variation,
            PromptVariationType.CONTEXT_ENRICHMENT: self._generate_context_variation,
            PromptVariationType.TEMPERATURE_ADJUSTMENT: self._generate_temperature_variation
        }
    
    async def initialize(self) -> None:
        """Initialize the prompt tuning system."""
        self.logger.info("Initializing prompt tuning system")
        
        # Load saved prompts
        self._load_saved_prompts()
        
        # Subscribe to events
        event_bus.subscribe("llm_api_request", self.handle_api_request)
        event_bus.subscribe("llm_api_response", self.handle_api_response)
        event_bus.subscribe("model_performance", self.handle_performance_update)
        
        # Set up initial active prompts if needed
        for prompt_type in ["sentiment_analysis", "event_detection", "impact_assessment"]:
            if prompt_type not in self.active_prompts and prompt_type in self.prompt_versions:
                # Use the best version as active
                best_version = self._select_best_prompt(prompt_type)
                if best_version:
                    self.active_prompts[prompt_type] = best_version
                    self.logger.info(f"Set active prompt for {prompt_type}: Version {best_version.version}")
    
    async def _load_saved_prompts(self) -> None:
        """Load saved prompts from storage."""
        try:
            # Check each prompt type directory
            for prompt_type in ["sentiment_analysis", "event_detection", "impact_assessment"]:
                type_path = os.path.join(self.storage_path, prompt_type)
                
                if not os.path.exists(type_path):
                    os.makedirs(type_path, exist_ok=True)
                    continue
                
                # Load all version files
                versions = []
                for filename in os.listdir(type_path):
                    if not filename.endswith(".json"):
                        continue
                    
                    file_path = os.path.join(type_path, filename)
                    try:
                        with open(file_path, 'r') as f:
                            prompt_data = json.load(f)
                            versions.append(PromptVersion.from_dict(prompt_data))
                    except Exception as e:
                        self.logger.error(f"Error loading prompt file {file_path}: {str(e)}")
                
                if versions:
                    # Sort by version number
                    versions.sort(key=lambda x: x.version)
                    self.prompt_versions[prompt_type] = versions
                    self.logger.info(f"Loaded {len(versions)} prompt versions for {prompt_type}")
        
        except Exception as e:
            self.logger.error(f"Error loading saved prompts: {str(e)}")
    
    async def _save_prompt_version(self, prompt: PromptVersion) -> None:
        """Save a prompt version to storage.
        
        Args:
            prompt: The prompt version to save
        """
        try:
            # Ensure type directory exists
            type_path = os.path.join(self.storage_path, prompt.prompt_type)
            os.makedirs(type_path, exist_ok=True)
            
            # Save to file
            file_path = os.path.join(type_path, f"{prompt.prompt_id}.json")
            with open(file_path, 'w') as f:
                json.dump(prompt.to_dict(), f, indent=2)
            
            self.logger.debug(f"Saved prompt {prompt.prompt_id} to {file_path}")
        
        except Exception as e:
            self.logger.error(f"Error saving prompt {prompt.prompt_id}: {str(e)}")
    
    def register_default_prompts(self, default_templates: Dict[str, str]) -> None:
        """Register default prompts from the LLM service.
        
        Args:
            default_templates: Dictionary of default prompt templates
        """
        for prompt_type, template in default_templates.items():
            # Check if we already have this prompt type
            if prompt_type in self.prompt_versions and self.prompt_versions[prompt_type]:
                continue  # Already have versions for this type
            
            # Create a new prompt version
            prompt_id = f"{prompt_type}_{uuid.uuid4().hex[:8]}"
            prompt = PromptVersion(
                prompt_id=prompt_id,
                prompt_type=prompt_type,
                template=template,
                version=1,
                description=f"Default {prompt_type} prompt"
            )
            
            # Add to versions
            self.prompt_versions[prompt_type] = [prompt]
            
            # Set as active
            self.active_prompts[prompt_type] = prompt
            
            # Save to storage
            asyncio.create_task(self._save_prompt_version(prompt))
            
            self.logger.info(f"Registered default prompt for {prompt_type}")
    
    def get_prompt_template(self, prompt_type: str, experiment: bool = False) -> Tuple[str, Dict[str, Any]]:
        """Get a prompt template for the specified type.
        
        Args:
            prompt_type: Type of prompt to get
            experiment: Whether to potentially use an experimental variant
            
        Returns:
            Tuple of (template, metadata)
        """
        # If type doesn't exist or no experiment requested, use active prompt
        if prompt_type not in self.prompt_versions or not experiment or not self.tuning_enabled:
            if prompt_type in self.active_prompts:
                prompt = self.active_prompts[prompt_type]
                return prompt.template, {"prompt_id": prompt.prompt_id, "version": prompt.version}
            else:
                self.logger.warning(f"No prompt found for type {prompt_type}")
                return "", {}
        
        # Decide whether to use an experimental prompt
        if random.random() < self.experiment_ratio:
            # Use a random prompt variant (not the active one)
            variants = [p for p in self.prompt_versions[prompt_type] 
                      if p.prompt_id != self.active_prompts[prompt_type].prompt_id]
            
            if variants:
                # Prioritize variants with fewer usages
                weights = [1.0 / (p.usage_count + 1) for p in variants]
                # Normalize weights
                total_weight = sum(weights)
                normalized_weights = [w / total_weight for w in weights]
                
                # Select a variant
                selected = np.random.choice(variants, p=normalized_weights)
                
                self.logger.info(f"Selected experimental prompt {selected.prompt_id} for {prompt_type}")
                return selected.template, {
                    "prompt_id": selected.prompt_id, 
                    "version": selected.version,
                    "is_experiment": True
                }
            
        # Use the active prompt
        prompt = self.active_prompts[prompt_type]
        return prompt.template, {"prompt_id": prompt.prompt_id, "version": prompt.version}
    
    def track_prompt_usage(self, request_id: str, prompt_id: str, data: Dict[str, Any]) -> None:
        """Track a prompt being used for a request.
        
        Args:
            request_id: Unique ID for the request
            prompt_id: ID of the prompt being used
            data: Request data
        """
        self.prompts_in_use[request_id] = {
            "prompt_id": prompt_id,
            "start_time": time.time(),
            "data": data
        }
    
    async def handle_api_request(self, event: Event) -> None:
        """Handle an LLM API request event.
        
        Args:
            event: The API request event
        """
        data = event.data
        request_id = data.get("request_id")
        prompt_type = data.get("prompt_type")
        prompt_id = data.get("prompt_id")
        
        if not all([request_id, prompt_type, prompt_id]):
            return
        
        # Track this request
        self.track_prompt_usage(request_id, prompt_id, data)
    
    async def handle_api_response(self, event: Event) -> None:
        """Handle an LLM API response event.
        
        Args:
            event: The API response event
        """
        data = event.data
        request_id = data.get("request_id")
        success = data.get("success", False)
        
        if not request_id or request_id not in self.prompts_in_use:
            return
        
        # Get request data
        request_data = self.prompts_in_use[request_id]
        prompt_id = request_data["prompt_id"]
        start_time = request_data["start_time"]
        
        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000
        
        # Find the prompt version
        prompt_version = None
        for versions in self.prompt_versions.values():
            for version in versions:
                if version.prompt_id == prompt_id:
                    prompt_version = version
                    break
            if prompt_version:
                break
        
        if not prompt_version:
            self.logger.warning(f"Unknown prompt ID: {prompt_id}")
            return
        
        # Record usage
        prompt_version.record_usage(success, latency_ms)
        
        # Save updated version
        asyncio.create_task(self._save_prompt_version(prompt_version))
        
        # Clean up
        del self.prompts_in_use[request_id]
        
        # Periodically evaluate prompts
        current_time = time.time()
        if current_time - self.last_evaluation_time > self.evaluation_interval:
            self.last_evaluation_time = current_time
            asyncio.create_task(self.evaluate_prompts())
    
    async def handle_performance_update(self, event: Event) -> None:
        """Handle a model performance update event.
        
        Args:
            event: The performance update event
        """
        data = event.data
        prompt_id = data.get("prompt_id")
        accuracy = data.get("accuracy", 0.0)
        calibration_error = data.get("calibration_error", 0.0)
        
        if not prompt_id:
            return
        
        # Find the prompt version
        prompt_version = None
        for versions in self.prompt_versions.values():
            for version in versions:
                if version.prompt_id == prompt_id:
                    prompt_version = version
                    break
            if prompt_version:
                break
        
        if not prompt_version:
            self.logger.warning(f"Unknown prompt ID in performance update: {prompt_id}")
            return
        
        # Update metrics
        prompt_version.update_metrics(accuracy, calibration_error)
        
        # Save updated version
        asyncio.create_task(self._save_prompt_version(prompt_version))
    
    async def evaluate_prompts(self) -> None:
        """Evaluate all prompts and select the best versions."""
        self.logger.info("Evaluating prompt performance")
        
        for prompt_type, versions in self.prompt_versions.items():
            # Skip if no versions or only one version
            if not versions or len(versions) == 1:
                continue
            
            # Get the current active prompt
            current_active = self.active_prompts.get(prompt_type)
            
            # Find the best performing prompt
            best_prompt = self._select_best_prompt(prompt_type)
            
            if not best_prompt:
                continue
                
            # Check if we should switch
            if (current_active and best_prompt.prompt_id != current_active.prompt_id and
                    best_prompt.usage_count >= self.min_usage_optimize):
                
                # Calculate the performance improvement
                current_score = current_active.get_performance_score()
                best_score = best_prompt.get_performance_score()
                improvement = best_score - current_score
                
                # Only switch if there's a meaningful improvement
                if improvement > 0.05:  # 5% improvement threshold
                    self.active_prompts[prompt_type] = best_prompt
                    self.logger.info(
                        f"Switched active prompt for {prompt_type} from "
                        f"version {current_active.version} ({current_score:.3f}) to "
                        f"version {best_prompt.version} ({best_score:.3f})"
                    )
                    
                    # Publish event for the switch
                    event_bus.publish(
                        "prompt_version_switch",
                        {
                            "prompt_type": prompt_type,
                            "old_version": current_active.version,
                            "old_id": current_active.prompt_id,
                            "new_version": best_prompt.version,
                            "new_id": best_prompt.prompt_id,
                            "improvement": improvement
                        }
                    )
            
            # Check if we should generate new variations
            if self.auto_optimize and best_prompt.usage_count >= self.min_usage_optimize:
                await self.generate_prompt_variations(prompt_type, best_prompt)
    
    def _select_best_prompt(self, prompt_type: str) -> Optional[PromptVersion]:
        """Select the best performing prompt for a type.
        
        Args:
            prompt_type: The prompt type
            
        Returns:
            Best performing prompt or None
        """
        if prompt_type not in self.prompt_versions:
            return None
        
        versions = self.prompt_versions[prompt_type]
        if not versions:
            return None
        
        # Filter to versions with enough usage
        qualified_versions = [v for v in versions if v.usage_count >= self.min_usage_optimize]
        
        # If none have enough usage, return the one with the most usage
        if not qualified_versions:
            return max(versions, key=lambda v: v.usage_count)
        
        # Sort by performance score
        return max(qualified_versions, key=lambda v: v.get_performance_score())
    
    async def generate_prompt_variations(
        self, 
        prompt_type: str, 
        base_prompt: PromptVersion,
        variation_types: Optional[List[PromptVariationType]] = None
    ) -> List[PromptVersion]:
        """Generate variations of a prompt.
        
        Args:
            prompt_type: Type of prompt
            base_prompt: Base prompt to generate variations from
            variation_types: Specific variation types to generate
            
        Returns:
            List of generated prompt versions
        """
        # Check if we already have too many versions
        if (prompt_type in self.prompt_versions and 
                len(self.prompt_versions[prompt_type]) >= self.max_versions):
            self.logger.info(
                f"Maximum versions ({self.max_versions}) reached for {prompt_type}, "
                "not generating variations"
            )
            return []
        
        # If no specific types requested, randomly select 1-2 types
        if not variation_types:
            num_variations = random.randint(1, 2)
            variation_types = random.sample(list(PromptVariationType), num_variations)
        
        generated_versions = []
        current_versions = len(self.prompt_versions.get(prompt_type, []))
        
        for variation_type in variation_types:
            # Get the generator for this variation type
            generator = self.variation_generators.get(variation_type)
            if not generator:
                continue
            
            # Generate the variation
            try:
                new_template, description = generator(base_prompt.template)
                
                # Create a new prompt version
                prompt_id = f"{prompt_type}_{uuid.uuid4().hex[:8]}"
                prompt = PromptVersion(
                    prompt_id=prompt_id,
                    prompt_type=prompt_type,
                    template=new_template,
                    version=current_versions + len(generated_versions) + 1,
                    parent_id=base_prompt.prompt_id,
                    variation_type=variation_type,
                    description=description
                )
                
                # Save the new version
                await self._save_prompt_version(prompt)
                
                # Add to versions list
                if prompt_type not in self.prompt_versions:
                    self.prompt_versions[prompt_type] = []
                self.prompt_versions[prompt_type].append(prompt)
                
                generated_versions.append(prompt)
                
                self.logger.info(
                    f"Generated {variation_type.value} variation for {prompt_type}, "
                    f"version {prompt.version}"
                )
                
            except Exception as e:
                self.logger.error(f"Error generating {variation_type.value} variation: {str(e)}")
        
        return generated_versions
    
    def _generate_instruction_variation(self, template: str) -> Tuple[str, str]:
        """Generate a variation with adjusted instructions.
        
        Args:
            template: Original template
            
        Returns:
            Tuple of (new_template, description)
        """
        # Parse the template to find the instruction section
        lines = template.split('\n')
        instructions_start = None
        instructions_end = None
        
        for i, line in enumerate(lines):
            if line.strip().startswith("Instructions:"):
                instructions_start = i
            elif instructions_start is not None and line.strip().startswith("Your response must be"):
                instructions_end = i
                break
        
        if not instructions_start or not instructions_end:
            raise ValueError("Could not locate instructions section in template")
        
        # Original instructions
        original_instructions = lines[instructions_start + 1:instructions_end]
        
        # Generate variations based on the prompt type
        if "sentiment analysis" in template.lower():
            new_instructions = [
                "1. Analyze the text for explicit and implicit sentiment indicators.",
                "2. Consider financial jargon, crypto-specific terminology, and market context.",
                "3. Weigh recent events more heavily than older information.",
                "4. Distinguish between short-term sentiment and longer-term outlook.",
                "5. Evaluate the credibility and potential impact of the content.",
                "6. Provide clear reasoning for your sentiment assessment."
            ]
        elif "event detection" in template.lower():
            new_instructions = [
                "1. Identify potentially market-moving events with high precision.",
                "2. Analyze the credibility, novelty, and potential market impact.",
                "3. Categorize events by type (regulatory, technical, adoption, etc.).",
                "4. Distinguish between rumored events and confirmed developments.",
                "5. Identify which specific assets would be most affected.",
                "6. Estimate the timeline for market impact (immediate, hours, days, weeks)."
            ]
        elif "impact assessment" in template.lower():
            new_instructions = [
                "1. Analyze short-term and long-term market impact possibilities.",
                "2. Consider different market regimes (bull market, bear market, sideways).",
                "3. Predict likely directional moves and potential magnitude.",
                "4. Identify which market participants would be most affected.",
                "5. Consider potential second-order effects on related assets.",
                "6. Estimate the duration of potential impact with justification."
            ]
        else:
            # Generic enhancement
            new_instructions = original_instructions + [
                f"{len(original_instructions) + 1}. Provide a confidence assessment for your analysis.",
                f"{len(original_instructions) + 2}. Consider potential alternative interpretations."
            ]
        
        # Replace instructions in template
        new_lines = lines.copy()
        new_lines[instructions_start + 1:instructions_end] = new_instructions
        
        # Create new template
        new_template = '\n'.join(new_lines)
        
        return new_template, "Enhanced instructions for more comprehensive analysis"
    
    def _generate_format_adjustment(self, template: str) -> Tuple[str, str]:
        """Generate a variation with adjusted output format.
        
        Args:
            template: Original template
            
        Returns:
            Tuple of (new_template, description)
        """
        # Find the JSON format section
        start_marker = "Your response must be in the following JSON format:"
        end_marker = "}"
        
        start_idx = template.find(start_marker)
        if start_idx == -1:
            raise ValueError("Could not locate format section in template")
        
        start_idx += len(start_marker)
        json_section = template[start_idx:].strip()
        
        # Find the last closing brace
        brace_count = 0
        end_idx = None
        
        for i, char in enumerate(json_section):
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    end_idx = i
                    break
        
        if end_idx is None:
            raise ValueError("Could not parse JSON format in template")
        
        json_format = json_section[:end_idx + 1]
        
        # Determine what kind of format to create
        if "sentiment_value" in json_format:
            # Enhance sentiment format
            new_format = """
{
    "sentiment_value": <float between 0 and 1, where 0 is extremely bearish, 0.5 is neutral, and 1 is extremely bullish>,
    "direction": <"bullish", "bearish", or "neutral">,
    "confidence": <float between 0 and 1 indicating your confidence level>,
    "explanation": <brief explanation of your reasoning>,
    "key_points": <list of key points that influenced your assessment>,
    "uncertainty_factors": <list of factors that could change your assessment>,
    "time_horizon": <"short_term", "medium_term", or "long_term">,
    "strength": <"weak", "moderate", or "strong">
}"""
        elif "is_market_event" in json_format:
            # Enhance event format
            new_format = """
{
    "is_market_event": <true or false>,
    "event_type": <category of event or null if not an event>,
    "assets_affected": <list of assets likely to be affected or null>,
    "severity": <integer from 0-10 indicating potential market impact or 0 if not an event>,
    "credibility": <float between 0 and 1 indicating credibility of the source and information>,
    "propagation_speed": <"immediate", "hours", "days", or "weeks">,
    "market_sentiment_impact": <"positive", "negative", or "neutral">,
    "confidence_level": <float between 0 and 1 indicating confidence in this analysis>,
    "explanation": <brief explanation of your reasoning>
}"""
        elif "primary_impact_direction" in json_format:
            # Enhance impact format
            new_format = """
{
    "primary_impact_direction": <"positive", "negative", or "neutral">,
    "impact_magnitude": <float between 0 and 1 indicating the magnitude>,
    "confidence": <float between 0 and 1 indicating your confidence level>,
    "estimated_duration": <"hours", "days", "weeks", or "months">,
    "affected_assets": <dictionary of assets and their relative impact scores>,
    "market_conditions": <list of market conditions that would amplify or reduce the impact>,
    "reasoning": <brief explanation of your reasoning>,
    "risk_factors": <list of factors that could alter your assessment>,
    "opportunity_assessment": <assessment of potential trading opportunities>
}"""
        else:
            # Generic enhancement
            new_format = json_format.replace(
                "}",
                ',\n    "confidence_level": <float between 0 and 1 indicating confidence in this analysis>,\n    "additional_notes": <any additional relevant information>\n}'
            )
        
        # Replace in template
        new_template = template[:start_idx] + new_format + template[start_idx + len(json_format):]
        
        return new_template, "Enhanced output format with additional structured fields"
    
    def _generate_system_role_variation(self, template: str) -> Tuple[str, str]:
        """Generate a variation with an enhanced system role.
        
        Args:
            template: Original template
            
        Returns:
            Tuple of (new_template, description)
        """
        # Find the role definition (usually the first line)
        lines = template.split('\n')
        role_line = -1
        
        for i, line in enumerate(lines):
            # Look for lines that define the assistant's role
            if "You are" in line and i < 5:
                role_line = i
                break
        
        if role_line == -1:
            # Default to first line if no explicit role found
            role_line = 0
        
        # Determine the type of prompt
        if "sentiment analyzer" in template.lower():
            new_role = """You are an expert financial sentiment analyzer with deep knowledge of cryptocurrency markets, tokenomics, and blockchain technology. You have years of experience in quantitative finance, market psychology, and technical analysis. Your specialty is extracting nuanced sentiment signals from various information sources while filtering out noise and manipulation attempts."""
        elif "event detector" in template.lower():
            new_role = """You are an elite financial event detector with specialized expertise in cryptocurrency markets and blockchain technology. You have a proven track record of identifying market-moving events before they impact prices. Your analysis incorporates regulatory impacts, technological developments, market psychology, and on-chain metrics to provide high-precision event detection."""
        elif "impact assessor" in template.lower():
            new_role = """You are a world-class financial impact assessor specializing in cryptocurrency markets. Your analysis combines macroeconomic understanding, quantitative modeling, market microstructure knowledge, and blockchain expertise. You excel at predicting how diverse events propagate through markets across different timeframes and market conditions."""
        else:
            # Generic enhanced role
            new_role = lines[role_line] + " with advanced expertise in quantitative finance, cryptocurrency markets, and blockchain technology."
        
        # Replace the role line
        new_lines = lines.copy()
        new_lines[role_line] = new_role
        
        return '\n'.join(new_lines), "Enhanced system role with specialized expertise"
    
    def _generate_example_variation(self, template: str) -> Tuple[str, str]:
        """Generate a variation with added examples.
        
        Args:
            template: Original template
            
        Returns:
            Tuple of (new_template, description)
        """
        # Find where to insert examples (after instructions, before JSON format)
        lines = template.split('\n')
        insert_point = None
        
        for i, line in enumerate(lines):
            if line.strip().startswith("Your response must be"):
                insert_point = i
                break
        
        if insert_point is None:
            # Try to find the last instruction line
            for i, line in enumerate(lines):
                if line.strip().startswith(str(i)) and i > 3:
                    insert_point = i + 1
                    break
        
        if insert_point is None:
            # Default to before the last 10 lines
            insert_point = max(0, len(lines) - 10)
        
        # Determine what kind of examples to add
        if "sentiment analysis" in template.lower():
            examples = """
Example Analysis:
Text: "Bitcoin just broke the $60,000 resistance level with strong volume. This could signal the beginning of a new uptrend, especially with increased institutional interest."
Analysis:
{
    "sentiment_value": 0.8,
    "direction": "bullish",
    "confidence": 0.75,
    "explanation": "Breaking a key resistance level with strong volume is typically a bullish signal, and institutional interest adds credibility",
    "key_points": ["Price broke key resistance", "Strong volume", "Institutional interest"]
}

Text: "Regulatory concerns continue to mount as three more countries consider restricting cryptocurrency trading. This uncertainty may drive volatility in the coming weeks."
Analysis:
{
    "sentiment_value": 0.3,
    "direction": "bearish",
    "confidence": 0.65,
    "explanation": "Regulatory restrictions typically create bearish pressure due to uncertainty and reduced market access",
    "key_points": ["Regulatory restrictions", "Multiple countries involved", "Expected volatility"]
}

"""
        elif "event detection" in template.lower():
            examples = """
Example Analysis:
Text: "BREAKING: SEC approves first spot Bitcoin ETF, trading to begin next week according to official documents."
Analysis:
{
    "is_market_event": true,
    "event_type": "regulatory_approval",
    "assets_affected": ["BTC", "ETH", "crypto_exchange_tokens"],
    "severity": 9,
    "credibility": 0.95,
    "propagation_speed": "days",
    "explanation": "SEC ETF approval is a major regulatory milestone that significantly improves institutional access to BTC"
}

Text: "Whale alert: 10,000 BTC moved from unknown wallet to Binance in the last hour."
Analysis:
{
    "is_market_event": true,
    "event_type": "large_transfer",
    "assets_affected": ["BTC"],
    "severity": 6,
    "credibility": 0.9,
    "propagation_speed": "immediate",
    "explanation": "Large transfers to exchanges often precede selling pressure, creating short-term volatility"
}

"""
        elif "impact assessment" in template.lower():
            examples = """
Example Assessment:
Event: "Central Bank announces 50 basis point interest rate hike, exceeding market expectations of 25 points"
Market Context: "Risk assets including cryptocurrencies have been in an uptrend for the past 3 months"
Assessment:
{
    "primary_impact_direction": "negative",
    "impact_magnitude": 0.7,
    "confidence": 0.8,
    "estimated_duration": "days",
    "affected_assets": {"BTC": -0.6, "ETH": -0.65, "DeFi tokens": -0.8},
    "reasoning": "Higher than expected interest rates typically reduce liquidity for risk assets, with DeFi particularly vulnerable",
    "risk_factors": ["Market already priced in some hike", "Strong technical uptrend might limit downside"]
}

Event: "Major cryptocurrency exchange announces new institutional service with custody solutions for corporations"
Market Context: "Market in consolidation phase after recent 20% correction"
Assessment:
{
    "primary_impact_direction": "positive",
    "impact_magnitude": 0.5,
    "confidence": 0.7,
    "estimated_duration": "weeks",
    "affected_assets": {"BTC": 0.6, "ETH": 0.5, "Exchange token": 0.9},
    "reasoning": "Improved institutional access creates sustainable demand, with exchange tokens benefiting directly from new revenue stream",
    "risk_factors": ["Implementation timeline unclear", "Competing services exist", "Regulatory approval pending"]
}

"""
        else:
            # Generic examples
            examples = """
Example:
(Input would be shown here)

Expected output:
{
    (JSON format with realistic values would be shown here)
}

"""
        
        # Insert examples
        new_lines = lines.copy()
        new_lines.insert(insert_point, examples)
        
        return '\n'.join(new_lines), "Added concrete examples for better context"
    
    def _generate_context_variation(self, template: str) -> Tuple[str, str]:
        """Generate a variation with additional context.
        
        Args:
            template: Original template
            
        Returns:
            Tuple of (new_template, description)
        """
        # Find where to insert context (after role, before instructions)
        lines = template.split('\n')
        role_line = -1
        instructions_line = -1
        
        for i, line in enumerate(lines):
            if "You are" in line and i < 5:
                role_line = i
            if line.strip().startswith("Instructions:"):
                instructions_line = i
                break
        
        if role_line == -1:
            role_line = 0
            
        if instructions_line == -1:
            # Try to find where the Text: or Event: section begins
            for i, line in enumerate(lines):
                if line.strip() in ["Text:", "Event:"]:
                    instructions_line = i
                    break
        
        if instructions_line == -1:
            # Default to middle of template
            instructions_line = len(lines) // 2
        
        insert_point = role_line + 1
        
        # Determine what kind of context to add
        if "sentiment analysis" in template.lower():
            context = """
Important Context for Cryptocurrency Sentiment Analysis:
1. Market Cycles: Crypto markets tend to follow cyclical patterns including accumulation, markup, distribution, and markdown phases.
2. Market Psychology: Sentiment often reacts sharply to news and can exhibit significant crowd psychology effects.
3. Information Asymmetry: News might affect different market participants differently based on their level of technical knowledge.
4. Regulatory Impact: Regulatory news typically has outsized impact on market sentiment compared to other developments.
5. Network Effects: Positive developments related to adoption often compound due to network effects.
6. Technical Significance: Technical indicators should be weighted based on trading volume and timeframe.

"""
        elif "event detection" in template.lower():
            context = """
Key Event Categories in Cryptocurrency Markets:
1. Regulatory Events: Government regulations, legal decisions, and policy changes (high impact).
2. Technology Events: Protocol upgrades, security incidents, and infrastructure developments.
3. Adoption Events: Corporate adoption, institutional investment, and new use cases.
4. Market Structure Events: Exchange listings, delistings, and trading pair changes.
5. Macroeconomic Events: Inflation data, interest rate changes, and global economic shifts.
6. Tokenomics Events: Token burns, emissions changes, and supply/distribution changes.

Typical Event Propagation Patterns:
- Regulatory events: Initially sharp reaction, followed by gradual acceptance
- Technical events: Impact depends on complexity and scope of changes
- Adoption events: Often underestimated in short-term, overestimated in long-term

"""
        elif "impact assessment" in template.lower():
            context = """
Market Impact Factors to Consider:
1. Market Regime: Bull markets amplify positive news; bear markets amplify negative news.
2. Liquidity Conditions: Low liquidity periods experience higher volatility from events.
3. Market Positioning: Crowded trades create asymmetric reactions when challenged.
4. Event Priming: Markets react differently to events that confirm or contradict prevailing narratives.
5. Cross-market Effects: Consider correlations with traditional markets when relevant.
6. Information Diffusion: Technical developments take longer to be fully priced in than simple news.
7. Stakeholder Impacts: Consider how the event affects different market participants (retail, institutional, miners, developers).

"""
        else:
            # Generic context
            context = """
Additional Context:
- Remember to consider both on-chain and off-chain factors in your analysis
- Factor in the credibility and track record of information sources
- Consider potential conflicts of interest in the information provided
- Distinguish between speculative information and confirmed facts

"""
        
        # Insert context
        new_lines = lines.copy()
        new_lines.insert(insert_point, context)
        
        return '\n'.join(new_lines), "Added domain-specific context for better grounding"
    
    def _generate_temperature_adjustment(self, template: str) -> Tuple[str, str]:
        """Generate a variation with temperature adjustment instructions.
        
        Args:
            template: Original template
            
        Returns:
            Tuple of (new_template, description)
        """
        # This doesn't modify the template itself, but adds metadata
        # to adjust the temperature parameter when calling the API
        
        # Randomly select a new temperature based on task
        if "sentiment analysis" in template.lower():
            # Lower temperature for more consistent sentiment analysis
            new_temp = round(random.uniform(0.0, 0.3), 2)
            description = f"Reduced temperature ({new_temp}) for more consistent sentiment analysis"
        elif "event detection" in template.lower():
            # Very low temperature for factual detection
            new_temp = round(random.uniform(0.0, 0.2), 2)
            description = f"Minimum temperature ({new_temp}) for precise event detection"
        elif "impact assessment" in template.lower():
            # Slightly higher temperature for creative impact assessment
            new_temp = round(random.uniform(0.1, 0.4), 2)
            description = f"Moderate temperature ({new_temp}) for balanced impact assessment"
        else:
            # Default
            new_temp = 0.2
            description = f"Adjusted temperature ({new_temp}) for balanced precision and creativity"
        
        # Add metadata instruction to template
        template_with_metadata = template + f"\n\n<!-- METADATA: {{'temperature': {new_temp}}} -->"
        
        return template_with_metadata, description
    
    async def archive_unused_prompts(self) -> None:
        """Archive prompt versions that haven't been used recently."""
        current_time = datetime.utcnow()
        archive_threshold = timedelta(days=30)  # Archive if not used in 30 days
        
        for prompt_type, versions in list(self.prompt_versions.items()):
            # Skip if fewer than max_versions
            if len(versions) <= self.max_versions:
                continue
            
            # Identify active and unused versions
            active_id = self.active_prompts.get(prompt_type, None)
            active_id = active_id.prompt_id if active_id else None
            
            unused_versions = []
            for version in versions:
                # Skip the active version
                if version.prompt_id == active_id:
                    continue
                
                # Check last usage time
                if not version.last_used:
                    unused_versions.append(version)
                    continue
                
                try:
                    last_used = datetime.fromisoformat(version.last_used)
                    if current_time - last_used > archive_threshold:
                        unused_versions.append(version)
                except (ValueError, TypeError):
                    # If we can't parse the date, consider it unused
                    unused_versions.append(version)
            
            # Sort unused versions by performance (keep better ones)
            unused_versions.sort(key=lambda v: v.get_performance_score(), reverse=True)
            
            # Determine how many to archive
            to_archive = unused_versions[-(len(versions) - self.max_versions):]
            
            if not to_archive:
                continue
                
            # Archive selected versions
            for version in to_archive:
                # Move to archive directory
                src_path = os.path.join(self.storage_path, prompt_type, f"{version.prompt_id}.json")
                archive_dir = os.path.join(self.storage_path, "archive", prompt_type)
                os.makedirs(archive_dir, exist_ok=True)
                dst_path = os.path.join(archive_dir, f"{version.prompt_id}.json")
                
                try:
                    # Save to archive
                    with open(dst_path, 'w') as f:
                        json.dump(version.to_dict(), f, indent=2)
                    
                    # Remove from active storage
                    if os.path.exists(src_path):
                        os.remove(src_path)
                    
                    # Remove from versions list
                    self.prompt_versions[prompt_type] = [
                        v for v in self.prompt_versions[prompt_type] 
                        if v.prompt_id != version.prompt_id
                    ]
                    
                    self.logger.info(f"Archived prompt {version.prompt_id} (version {version.version}) for {prompt_type}")
                    
                except Exception as e:
                    self.logger.error(f"Error archiving prompt {version.prompt_id}: {str(e)}")
    
    def get_all_prompts(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get all prompt versions.
        
        Returns:
            Dictionary of prompt types to lists of prompt version dictionaries
        """
        result = {}
        
        for prompt_type, versions in self.prompt_versions.items():
            result[prompt_type] = [v.to_dict() for v in versions]
            
            # Mark active prompts
            active_id = self.active_prompts.get(prompt_type, None)
            active_id = active_id.prompt_id if active_id else None
            
            for version_dict in result[prompt_type]:
                version_dict["is_active"] = version_dict["prompt_id"] = = active_id                version_dict["is_active"] = version_dict["prompt_id"] = = active_id
        
        return result
    
    def generate_prompt_report(self) -> Dict[str, Any]:
        """Generate a report on prompt performance.
        
        Returns:
            Report data
        """
        report = {
            "timestamp": datetime.utcnow().isoformat(),
            "prompt_types": {},
            "total_prompts": 0,
            "active_experiments": len(self.experiment_ids),
            "tuning_enabled": self.tuning_enabled,
            "auto_optimize": self.auto_optimize
        }
        
        total_count = 0
        
        for prompt_type, versions in self.prompt_versions.items():
            type_report = {
                "version_count": len(versions),
                "versions": [],
                "best_version": None,
                "active_version": None
            }
            
            # Get active version
            active = self.active_prompts.get(prompt_type)
            if active:
                type_report["active_version"] = {
                    "id": active.prompt_id,
                    "version": active.version,
                    "performance_score": active.get_performance_score(),
                    "usage_count": active.usage_count
                }
            
            # Get best version
            best = self._select_best_prompt(prompt_type)
            if best:
                type_report["best_version"] = {
                    "id": best.prompt_id,
                    "version": best.version,
                    "performance_score": best.get_performance_score(),
                    "usage_count": best.usage_count
                }
            
            # Version summaries
            for version in versions:
                type_report["versions"].append({
                    "id": version.prompt_id,
                    "version": version.version,
                    "performance_score": version.get_performance_score(),
                    "usage_count": version.usage_count,
                    "success_rate": version.get_success_rate(),
                    "avg_latency": version.get_average_latency(),
                    "accuracy": version.accuracy,
                    "is_active": active and version.prompt_id == active.prompt_id
                })
            
            report["prompt_types"][prompt_type] = type_report
            total_count += len(versions)
        
        report["total_prompts"] = total_count
        
        return report


# Singleton instance
prompt_tuning_system = PromptTuningSystem()