"""
LLM Fine-Tuning Module for AI Trading Agent.

This module provides tools and functionality for fine-tuning language models
on financial domain knowledge, regulatory documentation, market commentary,
and specialized trading contexts.
"""

from .config import FineTuningConfig
from .data_preparation import FinancialDatasetPreparer
from .model_fine_tuner import ModelFineTuner
from .evaluation import ModelEvaluator
from .prompt_templates import FinancialPromptTemplate, get_prompt_template

__all__ = [
    'FineTuningConfig',
    'FinancialDatasetPreparer',
    'ModelFineTuner',
    'ModelEvaluator',
    'FinancialPromptTemplate',
    'get_prompt_template',
]
