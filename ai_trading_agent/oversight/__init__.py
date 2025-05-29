"""
AI Trading Agent - LLM Oversight Module

This module provides LLM-based oversight capabilities for the AI Trading Agent,
enabling autonomous market analysis, decision validation, and error analysis.
"""

from .llm_oversight import LLMOversight, OversightLevel, LLMProvider
from .oversight_integration import OversightManager
from .trading_orchestrator_adapter import TradingOversightAdapter, OrchestratorHookType
from .config import OversightConfig, get_config, ConfigSource

__all__ = [
    'LLMOversight',
    'OversightLevel',
    'LLMProvider',
    'OversightManager',
    'TradingOversightAdapter',
    'OrchestratorHookType',
    'OversightConfig',
    'get_config',
    'ConfigSource'
]
