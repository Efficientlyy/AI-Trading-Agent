"""
Financial prompt templates for LLM fine-tuning and inference.

This module provides specialized prompt templates for financial domain tasks,
ensuring consistent prompting patterns for both training and inference.
"""
import os
import json
import logging
from typing import Dict, List, Optional, Tuple, Union, Any, Set
from enum import Enum
import re

logger = logging.getLogger(__name__)


class PromptTaskType(Enum):
    """Types of prompting tasks for financial domain."""
    MARKET_ANALYSIS = "market_analysis"
    TRADING_STRATEGY = "trading_strategy"
    RISK_ASSESSMENT = "risk_assessment"
    PORTFOLIO_OPTIMIZATION = "portfolio_optimization"
    REGULATORY_COMPLIANCE = "regulatory_compliance"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    ANOMALY_DETECTION = "anomaly_detection"
    ASSET_VALUATION = "asset_valuation"
    MACROECONOMIC_ANALYSIS = "macroeconomic_analysis"
    EARNINGS_ANALYSIS = "earnings_analysis"
    GENERAL = "general"


class FinancialPromptTemplate:
    """
    Template for generating prompts for financial domain tasks.
    
    This class provides structured templates for various financial tasks,
    ensuring consistent prompting patterns for both training and inference.
    """
    
    def __init__(self, task_type: PromptTaskType = PromptTaskType.GENERAL):
        """
        Initialize prompt template for a specific task type.
        
        Args:
            task_type: Type of financial task for this template
        """
        self.task_type = task_type
        self.system_prompt = self._get_default_system_prompt()
        self.prompt_structure = self._get_default_prompt_structure()
        self.required_fields: Set[str] = set()
        self.optional_fields: Set[str] = set()
        self._init_field_requirements()
    
    def _init_field_requirements(self) -> None:
        """Initialize required and optional fields based on task type."""
        # Common required fields across most tasks
        self.required_fields = {"context"}
        
        # Task-specific required and optional fields
        if self.task_type == PromptTaskType.MARKET_ANALYSIS:
            self.required_fields.update({"market", "timeframe"})
            self.optional_fields = {"indicators", "events", "prior_analysis"}
            
        elif self.task_type == PromptTaskType.TRADING_STRATEGY:
            self.required_fields.update({"market", "timeframe", "risk_tolerance"})
            self.optional_fields = {"capital", "constraints", "prior_trades"}
            
        elif self.task_type == PromptTaskType.RISK_ASSESSMENT:
            self.required_fields.update({"portfolio", "risk_factors"})
            self.optional_fields = {"timeframe", "risk_metrics", "constraints"}
            
        elif self.task_type == PromptTaskType.PORTFOLIO_OPTIMIZATION:
            self.required_fields.update({"portfolio", "objectives"})
            self.optional_fields = {"constraints", "risk_tolerance", "timeframe"}
            
        elif self.task_type == PromptTaskType.REGULATORY_COMPLIANCE:
            self.required_fields.update({"regulations", "scenario"})
            self.optional_fields = {"jurisdiction", "entity_type", "prior_compliance"}
            
        elif self.task_type == PromptTaskType.SENTIMENT_ANALYSIS:
            self.required_fields.update({"text", "entities"})
            self.optional_fields = {"source", "date", "prior_sentiment"}
            
        elif self.task_type == PromptTaskType.ANOMALY_DETECTION:
            self.required_fields.update({"data", "metrics"})
            self.optional_fields = {"thresholds", "historical_anomalies", "context"}
            
        elif self.task_type == PromptTaskType.ASSET_VALUATION:
            self.required_fields.update({"asset", "valuation_method"})
            self.optional_fields = {"comparables", "historical_data", "assumptions"}
            
        elif self.task_type == PromptTaskType.MACROECONOMIC_ANALYSIS:
            self.required_fields.update({"indicators", "regions"})
            self.optional_fields = {"timeframe", "scenario", "historical_context"}
            
        elif self.task_type == PromptTaskType.EARNINGS_ANALYSIS:
            self.required_fields.update({"company", "earnings_data"})
            self.optional_fields = {"historical_earnings", "estimates", "sector_context"}
            
        else:  # PromptTaskType.GENERAL
            self.required_fields = {"context"}
            self.optional_fields = {"additional_context", "instructions"}
    
    def _get_default_system_prompt(self) -> str:
        """
        Get the default system prompt for the task type.
        
        Returns:
            Default system prompt
        """
        base_prompt = (
            "You are a specialized financial AI assistant with expertise in financial markets, "
            "trading strategies, risk assessment, and portfolio management. "
            "You provide clear, concise, and accurate financial analysis and recommendations "
            "while adhering to regulatory compliance requirements."
        )
        
        task_specific_additions = {
            PromptTaskType.MARKET_ANALYSIS: (
                "\n\nSpecialization: You excel at analyzing market conditions, identifying trends, "
                "and providing insights based on technical and fundamental indicators. "
                "Your analysis incorporates multiple timeframes and considers relevant market events."
            ),
            PromptTaskType.TRADING_STRATEGY: (
                "\n\nSpecialization: You excel at developing and evaluating trading strategies "
                "across various market conditions. You consider risk tolerance, capital constraints, "
                "and market conditions to recommend appropriate entry and exit points."
            ),
            PromptTaskType.RISK_ASSESSMENT: (
                "\n\nSpecialization: You excel at identifying and quantifying financial risks "
                "in portfolios and trading strategies. You analyze various risk factors including "
                "market risk, credit risk, liquidity risk, and operational risk."
            ),
            PromptTaskType.PORTFOLIO_OPTIMIZATION: (
                "\n\nSpecialization: You excel at optimizing investment portfolios based on "
                "specified objectives such as risk-adjusted returns, diversification, and "
                "sector allocation. You balance risk and return according to investment constraints."
            ),
            PromptTaskType.REGULATORY_COMPLIANCE: (
                "\n\nSpecialization: You excel at navigating financial regulations and compliance "
                "requirements across various jurisdictions. You provide guidance on regulatory "
                "implications for trading activities and investment strategies."
            ),
            PromptTaskType.SENTIMENT_ANALYSIS: (
                "\n\nSpecialization: You excel at analyzing sentiment in financial texts, "
                "news articles, social media, and market commentary. You identify sentiment "
                "towards specific entities and assess potential market impact."
            ),
            PromptTaskType.ANOMALY_DETECTION: (
                "\n\nSpecialization: You excel at identifying unusual patterns and anomalies "
                "in financial data and market behavior. You distinguish between normal market "
                "volatility and potentially concerning irregularities."
            ),
            PromptTaskType.ASSET_VALUATION: (
                "\n\nSpecialization: You excel at valuing different types of financial assets "
                "using appropriate valuation methods such as DCF, comparable analysis, and "
                "multiple-based approaches. You consider relevant factors affecting valuation."
            ),
            PromptTaskType.MACROECONOMIC_ANALYSIS: (
                "\n\nSpecialization: You excel at analyzing macroeconomic indicators and their "
                "potential impact on financial markets. You assess economic conditions across "
                "regions and provide insights on economic trends and policy implications."
            ),
            PromptTaskType.EARNINGS_ANALYSIS: (
                "\n\nSpecialization: You excel at analyzing corporate earnings reports, "
                "identifying key metrics, and assessing performance relative to expectations. "
                "You evaluate earnings quality and provide insights on future earnings potential."
            )
        }
        
        # Add task-specific prompt if available
        if self.task_type in task_specific_additions:
            base_prompt += task_specific_additions[self.task_type]
        
        # Add compliance disclaimer
        base_prompt += (
            "\n\nIMPORTANT: Always include appropriate disclaimers about financial advice. "
            "Clearly state that your analysis is not a recommendation to buy or sell securities "
            "and that all investments involve risk. Recommend consulting with a financial advisor "
            "before making investment decisions."
        )
        
        return base_prompt
    
    def _get_default_prompt_structure(self) -> str:
        """
        Get the default prompt structure for the task type.
        
        Returns:
            Default prompt structure with placeholders
        """
        # Common structure elements
        common_structure = "# {task_name}\n\n"
        
        # Task-specific structures
        task_structures = {
            PromptTaskType.MARKET_ANALYSIS: (
                "## Market: {market}\n"
                "## Timeframe: {timeframe}\n"
                "## Relevant Indicators: {indicators}\n"
                "## Recent Events: {events}\n"
                "## Previous Analysis: {prior_analysis}\n\n"
                "## Context:\n{context}\n\n"
                "Analyze the current market conditions and provide insights on potential "
                "market direction, key support/resistance levels, and notable patterns. "
                "Identify significant technical and fundamental factors that could impact "
                "the market in the specified timeframe."
            ),
            PromptTaskType.TRADING_STRATEGY: (
                "## Market: {market}\n"
                "## Timeframe: {timeframe}\n"
                "## Risk Tolerance: {risk_tolerance}\n"
                "## Available Capital: {capital}\n"
                "## Trading Constraints: {constraints}\n"
                "## Recent Trades: {prior_trades}\n\n"
                "## Context:\n{context}\n\n"
                "Develop a trading strategy for the specified market and timeframe that "
                "aligns with the risk tolerance. Include specific entry and exit criteria, "
                "position sizing recommendations, and risk management guidelines. Consider "
                "current market conditions and potential scenarios that could affect the strategy."
            ),
            PromptTaskType.RISK_ASSESSMENT: (
                "## Portfolio: {portfolio}\n"
                "## Risk Factors: {risk_factors}\n"
                "## Assessment Timeframe: {timeframe}\n"
                "## Risk Metrics: {risk_metrics}\n"
                "## Constraints: {constraints}\n\n"
                "## Context:\n{context}\n\n"
                "Conduct a comprehensive risk assessment of the portfolio considering the "
                "specified risk factors. Quantify potential risks and their likelihood. "
                "Analyze portfolio vulnerabilities to market shocks, liquidity events, "
                "and other risk scenarios. Suggest risk mitigation measures if appropriate."
            ),
            PromptTaskType.PORTFOLIO_OPTIMIZATION: (
                "## Current Portfolio: {portfolio}\n"
                "## Optimization Objectives: {objectives}\n"
                "## Constraints: {constraints}\n"
                "## Risk Tolerance: {risk_tolerance}\n"
                "## Investment Horizon: {timeframe}\n\n"
                "## Context:\n{context}\n\n"
                "Optimize the portfolio based on the specified objectives while adhering to "
                "the constraints. Recommend adjustments to asset allocation, sector exposure, "
                "or specific holdings. Explain how the optimized portfolio improves upon the "
                "current portfolio in terms of the stated objectives."
            ),
            PromptTaskType.REGULATORY_COMPLIANCE: (
                "## Applicable Regulations: {regulations}\n"
                "## Scenario: {scenario}\n"
                "## Jurisdiction: {jurisdiction}\n"
                "## Entity Type: {entity_type}\n"
                "## Compliance History: {prior_compliance}\n\n"
                "## Context:\n{context}\n\n"
                "Analyze the regulatory compliance implications of the given scenario. "
                "Identify potential compliance issues and requirements. Suggest appropriate "
                "compliance measures and documentation needs. Consider jurisdictional "
                "differences and entity-specific regulations."
            ),
            PromptTaskType.SENTIMENT_ANALYSIS: (
                "## Text for Analysis: {text}\n"
                "## Entities of Interest: {entities}\n"
                "## Source: {source}\n"
                "## Date: {date}\n"
                "## Prior Sentiment: {prior_sentiment}\n\n"
                "## Context:\n{context}\n\n"
                "Analyze the sentiment expressed in the text towards the specified entities. "
                "Identify positive, negative, or neutral sentiments and their intensity. "
                "Extract key opinions or claims about the entities. Assess how this sentiment "
                "compares to prior sentiment if provided."
            ),
            PromptTaskType.ANOMALY_DETECTION: (
                "## Data Series: {data}\n"
                "## Metrics of Interest: {metrics}\n"
                "## Detection Thresholds: {thresholds}\n"
                "## Historical Anomalies: {historical_anomalies}\n\n"
                "## Context:\n{context}\n\n"
                "Analyze the data to identify potential anomalies in the specified metrics. "
                "Determine if any values exceed normal thresholds or exhibit unusual patterns. "
                "Compare to historical anomalies if provided. Explain the nature of any detected "
                "anomalies and their potential significance."
            ),
            PromptTaskType.ASSET_VALUATION: (
                "## Asset: {asset}\n"
                "## Valuation Method: {valuation_method}\n"
                "## Comparable Assets: {comparables}\n"
                "## Historical Data: {historical_data}\n"
                "## Key Assumptions: {assumptions}\n\n"
                "## Context:\n{context}\n\n"
                "Conduct a valuation of the specified asset using the indicated method. "
                "Consider comparable assets, historical data, and key assumptions. "
                "Provide a range of reasonable valuations and explain the factors that "
                "most significantly impact the valuation."
            ),
            PromptTaskType.MACROECONOMIC_ANALYSIS: (
                "## Economic Indicators: {indicators}\n"
                "## Regions: {regions}\n"
                "## Analysis Timeframe: {timeframe}\n"
                "## Scenario: {scenario}\n"
                "## Historical Context: {historical_context}\n\n"
                "## Context:\n{context}\n\n"
                "Analyze the current macroeconomic conditions based on the specified indicators "
                "and regions. Identify economic trends and potential shifts. Assess implications "
                "for financial markets and asset classes. Consider policy developments and their "
                "potential impact."
            ),
            PromptTaskType.EARNINGS_ANALYSIS: (
                "## Company: {company}\n"
                "## Current Earnings Data: {earnings_data}\n"
                "## Historical Earnings: {historical_earnings}\n"
                "## Analyst Estimates: {estimates}\n"
                "## Sector Context: {sector_context}\n\n"
                "## Context:\n{context}\n\n"
                "Analyze the company's earnings performance relative to expectations and "
                "historical results. Identify key drivers of performance and significant "
                "changes. Assess earnings quality and sustainability. Consider sector trends "
                "and competitive positioning in the analysis."
            ),
            PromptTaskType.GENERAL: (
                "## Context:\n{context}\n\n"
                "## Additional Information:\n{additional_context}\n\n"
                "## Instructions:\n{instructions}\n\n"
                "Provide a comprehensive analysis based on the given context and instructions. "
                "Consider all relevant financial factors and implications. Structure your response "
                "to address the specific questions or requirements indicated."
            )
        }
        
        # Get the structure for this task type or use general if not found
        task_structure = task_structures.get(
            self.task_type, 
            task_structures[PromptTaskType.GENERAL]
        )
        
        # Combine common and task-specific structure
        return common_structure + task_structure
    
    def set_system_prompt(self, system_prompt: str) -> 'FinancialPromptTemplate':
        """
        Set a custom system prompt.
        
        Args:
            system_prompt: Custom system prompt
            
        Returns:
            Self for method chaining
        """
        self.system_prompt = system_prompt
        return self
    
    def set_prompt_structure(self, prompt_structure: str) -> 'FinancialPromptTemplate':
        """
        Set a custom prompt structure.
        
        Args:
            prompt_structure: Custom prompt structure with placeholders
            
        Returns:
            Self for method chaining
        """
        self.prompt_structure = prompt_structure
        return self
    
    def format(self, **kwargs) -> Dict[str, str]:
        """
        Format the prompt template with provided values.
        
        Args:
            **kwargs: Values for placeholders in the template
            
        Returns:
            Dictionary with system_prompt and user_prompt
        """
        # Check for required fields
        missing_fields = self.required_fields - set(kwargs.keys())
        if missing_fields:
            missing_list = ", ".join(missing_fields)
            raise ValueError(f"Missing required fields for {self.task_type.value} prompt: {missing_list}")
        
        # Set task name if not provided
        if "task_name" not in kwargs:
            kwargs["task_name"] = self.task_type.value.replace("_", " ").title()
        
        # Fill in empty values for optional fields
        for field in self.optional_fields:
            if field not in kwargs:
                kwargs[field] = "N/A"
        
        # Format the prompt structure
        try:
            user_prompt = self.prompt_structure.format(**kwargs)
        except KeyError as e:
            # Handle missing placeholders gracefully
            logger.warning(f"Missing placeholder in prompt structure: {e}")
            # Attempt to remove the missing placeholder
            missing_field = str(e).strip("'")
            placeholder_pattern = r"\{" + missing_field + r"\}|\{" + missing_field + r":[^}]+\}"
            modified_structure = re.sub(placeholder_pattern, "", self.prompt_structure)
            # Try again with the modified structure
            user_prompt = modified_structure.format(**{k: v for k, v in kwargs.items() if k != missing_field})
        
        # Remove any empty lines that might result from optional fields
        user_prompt = re.sub(r'\n{3,}', '\n\n', user_prompt)
        
        return {
            "system_prompt": self.system_prompt,
            "user_prompt": user_prompt
        }
    
    def save(self, path: str) -> None:
        """
        Save the prompt template to a JSON file.
        
        Args:
            path: Path to save the template
        """
        template_data = {
            "task_type": self.task_type.value,
            "system_prompt": self.system_prompt,
            "prompt_structure": self.prompt_structure,
            "required_fields": list(self.required_fields),
            "optional_fields": list(self.optional_fields)
        }
        
        with open(path, "w") as f:
            json.dump(template_data, f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'FinancialPromptTemplate':
        """
        Load a prompt template from a JSON file.
        
        Args:
            path: Path to the template file
            
        Returns:
            Loaded template
        """
        with open(path, "r") as f:
            template_data = json.load(f)
        
        # Create instance with the task type
        template = cls(PromptTaskType(template_data["task_type"]))
        
        # Set loaded data
        template.set_system_prompt(template_data["system_prompt"])
        template.set_prompt_structure(template_data["prompt_structure"])
        template.required_fields = set(template_data["required_fields"])
        template.optional_fields = set(template_data["optional_fields"])
        
        return template
    
    @staticmethod
    def list_available_tasks() -> List[str]:
        """
        List all available task types.
        
        Returns:
            List of task type names
        """
        return [task_type.value for task_type in PromptTaskType]


def get_prompt_template(task_type_name: str) -> FinancialPromptTemplate:
    """
    Get a prompt template for a specific task by name.
    
    Args:
        task_type_name: Name of the task type
        
    Returns:
        Prompt template for the task
    """
    try:
        task_type = PromptTaskType(task_type_name)
        return FinancialPromptTemplate(task_type)
    except ValueError:
        logger.warning(f"Unknown task type: {task_type_name}. Using GENERAL template.")
        return FinancialPromptTemplate(PromptTaskType.GENERAL)
