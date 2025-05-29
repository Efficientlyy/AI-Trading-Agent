"""
AI Trading Agent - Agent Subpackage
------------------------------------

This subpackage contains the core components for the multi-agent system,
including base agent definitions, specific agent implementations, and the
orchestrator responsible for managing agent interactions and data flow.

Key modules:
- agent_definitions: Defines `BaseAgent`, `AgentRole`, `AgentStatus`, and
  example specialized agent classes like `SentimentAnalysisAgent`,
  `TechnicalAnalysisAgent`, `DecisionAgent`, and `ExecutionLayerAgent`.
- trading_orchestrator: Defines the `TradingOrchestrator` class, which
  manages the lifecycle and interactions of all registered agents.

This structure is based on the architecture outlined in the
`docs/agent_flow_architecture.md` document.
"""

from .agent_definitions import (
    AgentRole,
    AgentStatus,
    BaseAgent,
    SentimentAnalysisAgent,
    TechnicalAnalysisAgent,
    NewsEventAgent, # Added NewsEventAgent
    FundamentalAnalysisAgent, # Added FundamentalAnalysisAgent
    DecisionAgent,
    ExecutionLayerAgent
)
from .trading_orchestrator import TradingOrchestrator

__all__ = [
    "AgentRole",
    "AgentStatus",
    "BaseAgent",
    "SentimentAnalysisAgent",
    "TechnicalAnalysisAgent",
    "NewsEventAgent",
    "FundamentalAnalysisAgent", # Added FundamentalAnalysisAgent
    "DecisionAgent",
    "ExecutionLayerAgent",
    "TradingOrchestrator",
]
