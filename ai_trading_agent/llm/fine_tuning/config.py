"""
Configuration settings for LLM fine-tuning.

This module defines the configuration parameters for fine-tuning language
models on financial domain data, including model selection, training
parameters, and dataset specifications.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any
from enum import Enum
import os
import json


class ModelProvider(Enum):
    """Supported model providers for fine-tuning."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    AZURE_OPENAI = "azure_openai"
    HUGGING_FACE = "hugging_face"
    LOCAL = "local"


class FineTuningMethod(Enum):
    """Supported methods for fine-tuning."""
    FULL_FINE_TUNING = "full_fine_tuning"
    PEFT = "parameter_efficient_fine_tuning"
    LORA = "low_rank_adaptation"
    QLoRA = "quantized_low_rank_adaptation"
    PROMPT_TUNING = "prompt_tuning"
    INSTRUCTION_TUNING = "instruction_tuning"


class DatasetType(Enum):
    """Types of datasets for fine-tuning."""
    REGULATORY_DOCS = "regulatory_documents"
    MARKET_COMMENTARY = "market_commentary"
    FINANCIAL_NEWS = "financial_news"
    SEC_FILINGS = "sec_filings"
    EARNINGS_CALLS = "earnings_calls"
    TRADING_SCENARIOS = "trading_scenarios"
    MIXED = "mixed"


@dataclass
class FineTuningConfig:
    """Configuration for LLM fine-tuning."""
    
    # Model configuration
    model_provider: ModelProvider = ModelProvider.OPENAI
    base_model: str = "gpt-3.5-turbo"
    fine_tuning_method: FineTuningMethod = FineTuningMethod.INSTRUCTION_TUNING
    
    # API configuration
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    api_version: Optional[str] = None
    org_id: Optional[str] = None
    
    # Dataset configuration
    dataset_types: List[DatasetType] = field(default_factory=lambda: [DatasetType.MIXED])
    dataset_paths: Dict[DatasetType, str] = field(default_factory=dict)
    validation_split: float = 0.1
    test_split: float = 0.1
    
    # Training parameters
    batch_size: int = 4
    num_epochs: int = 3
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_steps: int = 500
    max_steps: Optional[int] = None
    gradient_accumulation_steps: int = 1
    
    # LoRA specific parameters (if applicable)
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    
    # Resource constraints
    max_gpu_memory: Optional[str] = None
    fp16: bool = True
    bf16: bool = False
    
    # Output configuration
    output_dir: str = "./fine_tuned_models"
    save_steps: int = 1000
    eval_steps: int = 500
    logging_steps: int = 100
    
    # Domain-specific parameters
    financial_domains: List[str] = field(
        default_factory=lambda: [
            "equity_trading",
            "forex",
            "commodities",
            "fixed_income",
            "crypto",
            "derivatives"
        ]
    )
    
    # Advanced configurations
    advanced_config: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate and initialize configuration after creation."""
        # Set API key from environment if not provided
        if self.api_key is None:
            provider_env_vars = {
                ModelProvider.OPENAI: "OPENAI_API_KEY",
                ModelProvider.ANTHROPIC: "ANTHROPIC_API_KEY", 
                ModelProvider.AZURE_OPENAI: "AZURE_OPENAI_API_KEY",
                ModelProvider.HUGGING_FACE: "HUGGINGFACE_API_KEY",
            }
            
            env_var = provider_env_vars.get(self.model_provider)
            if env_var and env_var in os.environ:
                self.api_key = os.environ[env_var]
        
        # Set appropriate default base models per provider if not specified
        if self.base_model == "gpt-3.5-turbo":
            if self.model_provider == ModelProvider.ANTHROPIC:
                self.base_model = "claude-2"
            elif self.model_provider == ModelProvider.HUGGING_FACE:
                self.base_model = "mistralai/Mistral-7B-v0.1"
    
    def save(self, path: str) -> None:
        """
        Save the configuration to a JSON file.
        
        Args:
            path: Path to save the configuration
        """
        # Convert enums to strings
        config_dict = {
            "model_provider": self.model_provider.value,
            "base_model": self.base_model,
            "fine_tuning_method": self.fine_tuning_method.value,
            "api_base": self.api_base,
            "api_version": self.api_version,
            "org_id": self.org_id,
            "dataset_types": [dt.value for dt in self.dataset_types],
            "dataset_paths": {dt.value: path for dt, path in self.dataset_paths.items()},
            "validation_split": self.validation_split,
            "test_split": self.test_split,
            "batch_size": self.batch_size,
            "num_epochs": self.num_epochs,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "warmup_steps": self.warmup_steps,
            "max_steps": self.max_steps,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "max_gpu_memory": self.max_gpu_memory,
            "fp16": self.fp16,
            "bf16": self.bf16,
            "output_dir": self.output_dir,
            "save_steps": self.save_steps,
            "eval_steps": self.eval_steps,
            "logging_steps": self.logging_steps,
            "financial_domains": self.financial_domains,
            "advanced_config": self.advanced_config
        }
        
        # Save to file (without API key for security)
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'FineTuningConfig':
        """
        Load a configuration from a JSON file.
        
        Args:
            path: Path to the configuration file
            
        Returns:
            Loaded configuration object
        """
        with open(path, 'r') as f:
            config_dict = json.load(f)
        
        # Convert string values back to enums
        config_dict["model_provider"] = ModelProvider(config_dict["model_provider"])
        config_dict["fine_tuning_method"] = FineTuningMethod(config_dict["fine_tuning_method"])
        config_dict["dataset_types"] = [DatasetType(dt) for dt in config_dict["dataset_types"]]
        
        # Convert dataset paths
        dataset_paths = {}
        for dt_str, path in config_dict.get("dataset_paths", {}).items():
            dataset_paths[DatasetType(dt_str)] = path
        config_dict["dataset_paths"] = dataset_paths
        
        return cls(**config_dict)
