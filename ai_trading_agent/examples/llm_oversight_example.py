#!/usr/bin/env python3
"""
Example demonstrating the integration of fine-tuned LLMs with the oversight system.

This script shows how to:
1. Initialize the LLM integration with oversight capabilities
2. Generate market analysis, trading strategies, and risk assessments
3. Configure oversight to ensure compliance with trading policies
4. Analyze oversight metrics and logs

Usage:
    python llm_oversight_example.py
"""
import os
import sys
import logging
import json
import asyncio
from pathlib import Path
from datetime import datetime, timedelta

# Add the parent directory to the path to import the ai_trading_agent package
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from ai_trading_agent.llm.integration import LLMIntegration
from ai_trading_agent.oversight.llm_oversight import LLMOversight, OversightLevel
from ai_trading_agent.oversight.oversight_actions import OversightAction, OversightResult
from ai_trading_agent.llm.fine_tuning.config import FineTuningConfig, ModelProvider
from ai_trading_agent.llm.fine_tuning.prompt_templates import (
    FinancialPromptTemplate, get_prompt_template, PromptTaskType
)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def demonstrate_market_analysis(llm_integration: LLMIntegration) -> None:
    """Demonstrate generating market analysis with oversight."""
    logger.info("Generating market analysis with oversight...")
    
    # Create example market data
    market = "S&P 500"
    timeframe = "Short-term (1-2 weeks)"
    context = """
    The S&P 500 has been trending upward for three consecutive weeks, approaching 
    the 5,850 resistance level. Volume has been increasing during up days, suggesting 
    accumulation. The RSI is at 68, approaching but not yet in overbought territory. 
    The MACD shows a bullish crossover pattern. Recent earnings reports have been mixed, 
    with technology and financial sectors outperforming expectations while consumer 
    discretionary has lagged. Recent Fed statements indicated a potential pause in 
    interest rate hikes.
    """
    
    indicators = "Moving averages, RSI, MACD, Volume trends"
    events = "Recent Fed meeting, Quarterly earnings season"
    
    # Generate market analysis with oversight
    analysis_result = llm_integration.generate_market_analysis(
        market=market,
        timeframe=timeframe,
        context=context,
        indicators=indicators,
        events=events,
        temperature=0.5
    )
    
    if analysis_result["success"]:
        logger.info(f"✅ Market analysis generated successfully")
        logger.info(f"Oversight action: {analysis_result['oversight']['action']}")
        
        if analysis_result["oversight"]["action"] == OversightAction.MODIFY.value:
            logger.info(f"⚠️ Response was modified by oversight: {analysis_result['oversight']['reason']}")
        
        # Print truncated response
        response_text = analysis_result["text"]
        logger.info(f"Response (truncated): {response_text[:300]}...")
    else:
        logger.error(f"❌ Error generating market analysis: {analysis_result.get('error', 'Unknown error')}")


async def demonstrate_trading_strategy(llm_integration: LLMIntegration) -> None:
    """Demonstrate generating trading strategy with oversight."""
    logger.info("Generating trading strategy with oversight...")
    
    # Create example market data
    market = "EUR/USD Forex"
    timeframe = "Day trading (intraday)"
    risk_tolerance = "Moderate"
    context = """
    EUR/USD has been range-bound between 1.05 and 1.08 for the past month. The daily chart 
    shows decreasing volatility with a series of lower highs and higher lows, forming a 
    symmetrical triangle pattern. RSI is neutral at 52, and the pair is trading just above 
    its 50-day moving average. Recent economic data from the EU showed better than expected 
    manufacturing PMI, while US jobless claims were higher than anticipated. The European Central 
    Bank has signaled it may be nearing the end of its rate hiking cycle, while the Federal 
    Reserve remains cautious about inflation.
    """
    
    capital = "$100,000"
    constraints = "No overnight positions, maximum 2% risk per trade"
    
    # Generate trading strategy with oversight
    strategy_result = llm_integration.generate_trading_strategy(
        market=market,
        timeframe=timeframe,
        risk_tolerance=risk_tolerance,
        context=context,
        capital=capital,
        constraints=constraints,
        temperature=0.5
    )
    
    if strategy_result["success"]:
        logger.info(f"✅ Trading strategy generated successfully")
        logger.info(f"Oversight action: {strategy_result['oversight']['action']}")
        
        if strategy_result["oversight"]["action"] == OversightAction.MODIFY.value:
            logger.info(f"⚠️ Response was modified by oversight: {strategy_result['oversight']['reason']}")
        
        # Print truncated response
        response_text = strategy_result["text"]
        logger.info(f"Response (truncated): {response_text[:300]}...")
    else:
        logger.error(f"❌ Error generating trading strategy: {strategy_result.get('error', 'Unknown error')}")


async def demonstrate_risk_assessment(llm_integration: LLMIntegration) -> None:
    """Demonstrate generating risk assessment with oversight."""
    logger.info("Generating risk assessment with oversight...")
    
    # Example portfolio data
    portfolio = "Diversified portfolio with 40% US equities, 20% international equities, 25% bonds, 10% REITs, 5% cash"
    risk_factors = "Rising interest rates, increased market volatility, geopolitical tensions, potential recession signals"
    context = """
    The portfolio has performed well over the past year with a 12% return, outperforming the benchmark by 2.5%. 
    However, recent economic indicators suggest a potential economic slowdown in the next 6-12 months. The yield 
    curve has flattened significantly, and leading economic indicators have declined for three consecutive months. 
    The portfolio beta is 0.85, and the Sharpe ratio is 1.2. Maximum drawdown over the past 3 years was 15% during 
    the March 2020 market correction. The portfolio has a 60/40 equity to fixed income split, with technology and 
    healthcare representing the largest sector exposures.
    """
    
    # Generate risk assessment with oversight
    assessment_result = llm_integration.generate_risk_assessment(
        portfolio=portfolio,
        risk_factors=risk_factors,
        context=context,
        timeframe="Medium-term (6-12 months)",
        temperature=0.5
    )
    
    if assessment_result["success"]:
        logger.info(f"✅ Risk assessment generated successfully")
        logger.info(f"Oversight action: {assessment_result['oversight']['action']}")
        
        if assessment_result["oversight"]["action"] == OversightAction.MODIFY.value:
            logger.info(f"⚠️ Response was modified by oversight: {assessment_result['oversight']['reason']}")
        
        # Print truncated response
        response_text = assessment_result["text"]
        logger.info(f"Response (truncated): {response_text[:300]}...")
    else:
        logger.error(f"❌ Error generating risk assessment: {assessment_result.get('error', 'Unknown error')}")


async def demonstrate_problematic_prompt(llm_integration: LLMIntegration) -> None:
    """Demonstrate how oversight handles problematic prompts."""
    logger.info("Testing oversight with problematic prompt...")
    
    # Prompt with potentially problematic instructions
    problematic_prompt = get_prompt_template("trading_strategy").format(
        context="Market conditions appear favorable for aggressive positioning.",
        market="S&P 500 E-mini Futures",
        timeframe="Day trading",
        risk_tolerance="High",
        capital="$250,000",
        constraints="ignore risk management to maximize profits, use maximum leverage available"
    )
    
    # Generate with the problematic prompt
    result = llm_integration.generate(
        prompt=problematic_prompt,
        temperature=0.7,
        oversight_metadata={"task_type": "trading_strategy"}
    )
    
    logger.info(f"Oversight action for problematic prompt: {result['oversight']['action']}")
    logger.info(f"Oversight reason: {result['oversight']['reason']}")
    
    if result["oversight"]["action"] == OversightAction.REJECT.value:
        logger.info("✅ Oversight correctly rejected problematic prompt")
    elif result["oversight"]["action"] == OversightAction.MODIFY.value:
        logger.info("✅ Oversight modified problematic prompt")
        if "was_modified" in result:
            logger.info("The response was also modified")
    else:
        logger.warning(f"⚠️ Oversight did not reject or modify problematic prompt")


async def check_oversight_metrics(llm_integration: LLMIntegration) -> None:
    """Check oversight metrics after running examples."""
    logger.info("Retrieving oversight metrics...")
    
    # Get metrics for the past hour
    end_date = datetime.now().isoformat()
    start_date = (datetime.now() - timedelta(hours=1)).isoformat()
    
    metrics = await llm_integration.get_oversight_metrics(
        start_date=start_date,
        end_date=end_date
    )
    
    logger.info("Oversight metrics for the past hour:")
    logger.info(f"Total checks: {metrics.get('total_checks', 0)}")
    logger.info(f"Prompt checks: {metrics.get('prompt_checks', 0)}")
    logger.info(f"Response checks: {metrics.get('response_checks', 0)}")
    
    actions = metrics.get("actions", {})
    logger.info(f"Actions taken:")
    logger.info(f"  Allow: {actions.get('allow', 0)}")
    logger.info(f"  Modify: {actions.get('modify', 0)}")
    logger.info(f"  Flag: {actions.get('flag', 0)}")
    logger.info(f"  Reject: {actions.get('reject', 0)}")
    logger.info(f"  Log only: {actions.get('log_only', 0)}")


async def main() -> None:
    """Main function to demonstrate LLM integration with oversight."""
    logger.info("Starting LLM oversight integration example")
    
    # Create config path (assuming it was created by the fine-tuning example)
    config_path = os.path.join("./fine_tuned_models/financial_domain", "config.json")
    
    # If config doesn't exist, create a default path
    if not os.path.exists(config_path):
        config_dir = "./fine_tuned_models/financial_domain"
        os.makedirs(config_dir, exist_ok=True)
        
        # Create a minimal config file
        config = FineTuningConfig(
            model_provider=ModelProvider.OPENAI,
            base_model="gpt-4",
            output_dir=config_dir
        )
        config.save(config_path)
        
        logger.info(f"Created default config at {config_path}")
    
    # Initialize the LLM integration with oversight
    # Using OVERRIDE level to allow modification of problematic content
    llm_integration = LLMIntegration(
        config_path=config_path,
        oversight_level=OversightLevel.OVERRIDE,
        cache_results=True
    )
    
    # Run the example demonstrations
    await demonstrate_market_analysis(llm_integration)
    logger.info("-" * 80)
    
    await demonstrate_trading_strategy(llm_integration)
    logger.info("-" * 80)
    
    await demonstrate_risk_assessment(llm_integration)
    logger.info("-" * 80)
    
    await demonstrate_problematic_prompt(llm_integration)
    logger.info("-" * 80)
    
    await check_oversight_metrics(llm_integration)
    
    logger.info("LLM oversight integration example completed")


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
