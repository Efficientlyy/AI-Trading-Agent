"""
Custom prompts for the LLM Oversight system.

This module contains prompt templates for different oversight functions
such as market analysis, decision validation, and anomaly detection.
"""

from typing import Dict, Any, List, Optional
import json

# Prompt template for market analysis
MARKET_ANALYSIS_PROMPT = """
You are an expert financial analyst and market regime classifier specializing in algorithmic trading.
Analyze the provided market data and provide structured insights.

# Market Data
{market_data_summary}

{additional_context}

# Your Analysis Task
Provide a comprehensive market analysis with the following:
1. Classify the current market regime (bullish, bearish, neutral, volatile, or range-bound)
2. Identify key patterns or anomalies in price action and volume
3. Assess the overall market sentiment
4. Evaluate market divergences or confirmations
5. Analyze volatility characteristics
6. Highlight macro or sector-specific trends relevant to these assets
7. Identify any pending signals or patterns that could indicate regime transitions

# Format
Respond with a valid JSON object with the following structure:
{
    "regime": "string (one of: bullish, bearish, neutral, volatile, range-bound)",
    "confidence": float (0.0-1.0),
    "key_insights": [string, string, ...], (3-5 most important observations)
    "outlook": string (concise forward-looking assessment),
    "risk_factors": [string, string, ...] (2-3 key risks to consider)
}

Focus on objective data analysis, patterns, and quantifiable metrics. Be specific and precise.
"""

# Prompt template for trading decision validation
DECISION_VALIDATION_PROMPT = """
You are an expert trading oversight system responsible for validating algorithmic trading decisions.
Evaluate the following trading decision based on market context and trading logic.

# Trading Decision
{decision_json}

# Market Context
{context_json}

# Your Validation Task
Evaluate this trading decision with respect to:
1. Alignment with the current market regime and conditions
2. Risk management rules and position sizing
3. Timing of entry/exit based on price action
4. Consistency with the trading strategy's objective
5. Potential execution risks or slippage concerns
6. Current portfolio allocation and concentration
7. Technical and fundamental justification

# Response Format
Respond with a JSON object with the following structure:
{
    "action": string (one of: "approve", "reject", "modify"),
    "confidence": float (0.0-1.0),
    "reason": string (brief justification for your decision),
    "recommendation": string (if action is "modify" or there's room for improvement)
}

Focus on objective analysis based on market principles and trading best practices.
Be specific about why you approved, rejected, or suggested modifications.
"""

# Prompt template for anomaly detection
ANOMALY_DETECTION_PROMPT = """
You are an expert risk management system specializing in detecting trading anomalies and market irregularities.
Analyze the following trading data and identify any potential anomalies or outliers.

# Data for Analysis
{trading_data_json}

# Historical Patterns
{historical_patterns}

# Your Detection Task
Identify any anomalies or irregularities in the trading data, focusing on:
1. Unusual price movements relative to historical volatility
2. Volume anomalies indicating potential market manipulation
3. Correlation breakdowns between related assets
4. Unusual spread behavior or liquidity conditions
5. Order flow imbalances or irregular trade timing
6. Deviation from expected seasonal patterns
7. Statistical outliers in any key metrics

# Response Format
Respond with a JSON object with the following structure:
{
    "anomalies_detected": boolean,
    "confidence": float (0.0-1.0),
    "findings": [
        {
            "type": string (category of anomaly),
            "description": string (detailed explanation),
            "severity": string (one of: "low", "medium", "high", "critical"),
            "recommended_action": string (what action to take)
        }
    ],
    "summary": string (brief overall assessment)
}

Focus on statistical significance and comparative analysis.
Avoid false positives by considering alternative explanations.
"""

# Prompt template for strategy adjustment
STRATEGY_ADJUSTMENT_PROMPT = """
You are an expert trading strategy optimizer specializing in algorithmic trading.
Analyze the strategy performance data and suggest optimizations.

# Strategy Performance
{performance_data}

# Market Conditions
{market_conditions}

# Current Strategy Configuration
{strategy_config}

# Your Optimization Task
Recommend adjustments to improve the strategy's performance, focusing on:
1. Parameter optimization for current market conditions
2. Risk management improvements
3. Entry/exit timing refinements
4. Filtering criteria to reduce false signals
5. Adaptation to changing market regimes
6. Performance robustness across different conditions

# Response Format
Respond with a JSON object with the following structure:
{
    "adjustments": [
        {
            "parameter": string (name of parameter to adjust),
            "current_value": any (current value),
            "recommended_value": any (suggested value),
            "rationale": string (explanation for this change),
            "expected_impact": string (anticipated effect on performance)
        }
    ],
    "new_features": [
        {
            "name": string (name of new feature to add),
            "description": string (how to implement),
            "benefit": string (expected benefit)
        }
    ],
    "priority": string (which adjustment should be implemented first)
}

Make recommendations that are specific, measurable, and realistic.
Focus on data-driven insights rather than general trading advice.
"""


def format_market_data_summary(market_data: Dict[str, Any]) -> str:
    """
    Format market data for inclusion in prompts.
    
    Args:
        market_data: Dictionary containing market data for various symbols
        
    Returns:
        Formatted string summarizing the market data
    """
    summary_parts = []
    
    for symbol, data in market_data.items():
        # Check data structure to accommodate different formats
        if isinstance(data, dict) and 'close' in data:
            # If data contains lists of values
            if isinstance(data['close'], list):
                close_prices = data['close']
                n = len(close_prices)
                
                # Calculate some basic stats
                if n >= 2:
                    recent_return = (close_prices[-1] / close_prices[-2] - 1) * 100
                else:
                    recent_return = 0
                    
                if n >= 20:
                    period_return = (close_prices[-1] / close_prices[-20] - 1) * 100
                else:
                    period_return = (close_prices[-1] / close_prices[0] - 1) * 100
                
                # Get last price and some representative prices from the series
                last_price = close_prices[-1]
                
                # Format the summary for this symbol
                symbol_summary = (
                    f"{symbol}: Last price ${last_price:.2f}, "
                    f"1-day change: {recent_return:.2f}%, "
                    f"Period change: {period_return:.2f}%"
                )
                
                # Add volume info if available
                if 'volume' in data and isinstance(data['volume'], list) and len(data['volume']) > 0:
                    last_volume = data['volume'][-1]
                    symbol_summary += f", Volume: {last_volume:,.0f}"
                
                summary_parts.append(symbol_summary)
        
    return "\n".join(summary_parts)


def create_market_analysis_prompt(
    market_data: Dict[str, Any], 
    additional_context: Optional[str] = None
) -> str:
    """
    Create a prompt for market analysis.
    
    Args:
        market_data: Dictionary of market data by symbol
        additional_context: Optional additional context to include
        
    Returns:
        Formatted prompt string
    """
    market_data_summary = format_market_data_summary(market_data)
    
    if not additional_context:
        additional_context = """
# Additional Context
Consider the recent market volatility and sector trends when analyzing these assets.
"""
    
    return MARKET_ANALYSIS_PROMPT.format(
        market_data_summary=market_data_summary,
        additional_context=additional_context
    )


def create_decision_validation_prompt(
    decision: Dict[str, Any],
    context: Dict[str, Any]
) -> str:
    """
    Create a prompt for validating a trading decision.
    
    Args:
        decision: Trading decision dictionary
        context: Market and portfolio context
        
    Returns:
        Formatted prompt string
    """
    # Format decision as pretty JSON for readability
    decision_json = json.dumps(decision, indent=2)
    context_json = json.dumps(context, indent=2)
    
    return DECISION_VALIDATION_PROMPT.format(
        decision_json=decision_json,
        context_json=context_json
    )


def create_anomaly_detection_prompt(
    trading_data: Dict[str, Any],
    historical_patterns: str
) -> str:
    """
    Create a prompt for anomaly detection.
    
    Args:
        trading_data: Current trading data
        historical_patterns: Description of historical patterns
        
    Returns:
        Formatted prompt string
    """
    trading_data_json = json.dumps(trading_data, indent=2)
    
    return ANOMALY_DETECTION_PROMPT.format(
        trading_data_json=trading_data_json,
        historical_patterns=historical_patterns
    )


def create_strategy_adjustment_prompt(
    performance_data: Dict[str, Any],
    market_conditions: Dict[str, Any],
    strategy_config: Dict[str, Any]
) -> str:
    """
    Create a prompt for strategy adjustment recommendations.
    
    Args:
        performance_data: Strategy performance metrics
        market_conditions: Current market conditions
        strategy_config: Current strategy configuration
        
    Returns:
        Formatted prompt string
    """
    performance_data_str = json.dumps(performance_data, indent=2)
    market_conditions_str = json.dumps(market_conditions, indent=2)
    strategy_config_str = json.dumps(strategy_config, indent=2)
    
    return STRATEGY_ADJUSTMENT_PROMPT.format(
        performance_data=performance_data_str,
        market_conditions=market_conditions_str,
        strategy_config=strategy_config_str
    )
