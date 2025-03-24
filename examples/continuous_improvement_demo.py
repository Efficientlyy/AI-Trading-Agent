#!/usr/bin/env python
"""
Continuous Improvement System Demo.

This script demonstrates the automatic optimization capabilities of the
continuous improvement system for sentiment analysis.
"""

import asyncio
import logging
import json
import time
from datetime import datetime, timedelta
import random

from src.analysis_agents.sentiment.continuous_improvement import continuous_improvement_manager
from src.analysis_agents.sentiment.ab_testing import (
    ab_testing_framework, ExperimentType, ExperimentStatus,
    TargetingCriteria, VariantAssignmentStrategy
)
from src.analysis_agents.sentiment.llm_service import LLMService
from src.common.logging import get_logger


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = get_logger("examples", "continuous_improvement_demo")


async def setup_demo():
    """Set up the demo by initializing required services."""
    # Initialize the A/B testing framework
    await ab_testing_framework.initialize()
    
    # Enable and initialize the continuous improvement system
    continuous_improvement_manager.enabled = True
    continuous_improvement_manager.auto_implement = True
    await continuous_improvement_manager.initialize()
    
    # Initialize the LLM service
    llm_service = LLMService()
    await llm_service.initialize()
    
    return llm_service


async def simulate_sentiment_analysis(llm_service, iterations=20):
    """Simulate sentiment analysis requests to generate data for experiments.
    
    Args:
        llm_service: LLM service instance
        iterations: Number of iterations to run
    """
    logger.info(f"Simulating {iterations} sentiment analysis requests...")
    
    # Sample texts for sentiment analysis
    sample_texts = [
        "Bitcoin has broken above $75,000 for the first time ever, setting a new all-time high as institutional adoption continues to grow with multiple ETF approvals.",
        "Ethereum's transition to proof-of-stake has been delayed again, raising concerns about the platform's ability to scale and compete with newer blockchains.",
        "Regulatory uncertainty continues to plague the crypto market as another exchange faces scrutiny from authorities over its compliance procedures.",
        "The latest upgrade to the Solana network has significantly improved transaction speeds and reduced fees, attracting more developers to the platform.",
        "Bitcoin mining difficulty has reached an all-time high, potentially squeezing profit margins for smaller miners in the current market conditions.",
        "A new report suggests that central bank digital currencies could pose a threat to decentralized cryptocurrencies as governments accelerate their development plans.",
        "The latest DeFi protocol has attracted over $1 billion in total value locked within its first week, showing continued interest in decentralized finance despite recent market volatility.",
        "Market analysts predict sideways trading for major cryptocurrencies in the coming weeks as the market digests recent gains and awaits new catalysts.",
        "A previously unknown vulnerability in a popular wallet has been patched after the discovery of a potential exploit that could have put user funds at risk.",
        "Institutional investors continue to accumulate Bitcoin during price dips, according to on-chain data showing large wallet movements from exchanges to cold storage."
    ]
    
    for i in range(iterations):
        # Choose a random text
        text = random.choice(sample_texts)
        
        # Create a random context ID and symbol
        context_id = f"ctx_{random.randint(1000, 9999)}"
        symbol = random.choice(["BTC", "ETH", "SOL", "XRP"])
        
        logger.info(f"Analysis iteration {i+1} for context {context_id}, symbol {symbol}")
        
        # Analyze the sentiment
        try:
            result = await llm_service.analyze_sentiment(text)
            logger.info(f"Result direction: {result.get('direction', 'unknown')}, "
                        f"value: {result.get('sentiment_value', 0):.2f}, "
                        f"confidence: {result.get('confidence', 0):.2f}")
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
        
        # Small delay to avoid rate limits
        await asyncio.sleep(1)


async def generate_demo_experiment():
    """Generate a demo experiment for testing."""
    logger.info("Generating demo experiment...")
    
    # Create a prompt template experiment
    experiment = ab_testing_framework.create_experiment(
        name="Demo Prompt Template Test",
        description="Testing improved prompt instructions for sentiment analysis",
        experiment_type=ExperimentType.PROMPT_TEMPLATE,
        variants=[
            {
                "name": "Standard Template",
                "description": "Default template for sentiment analysis",
                "weight": 0.5,
                "config": {
                    "template": """
You are a financial sentiment analyzer specialized in cryptocurrency and blockchain markets.
Analyze the following text and determine the overall market sentiment.

Text:
{text}

Instructions:
1. Analyze the text for bullish, bearish, or neutral sentiment.
2. Consider financial jargon and crypto-specific terminology.
3. Evaluate the credibility and potential impact of the content.
4. Provide an explanation for your reasoning.

Your response must be in the following JSON format:
{
    "sentiment_value": <float between 0 and 1, where 0 is extremely bearish, 0.5 is neutral, and 1 is extremely bullish>,
    "direction": <"bullish", "bearish", or "neutral">,
    "confidence": <float between 0 and 1 indicating your confidence level>,
    "explanation": <brief explanation of your reasoning>,
    "key_points": <list of key points that influenced your assessment>
}
"""
                },
                "control": True
            },
            {
                "name": "Enhanced Template",
                "description": "Template with improved structure and additional context",
                "weight": 0.5,
                "config": {
                    "template": """
You are an expert cryptocurrency market analyst with deep knowledge of blockchain technology, tokenomics, and market psychology.

Context for Crypto Sentiment Analysis:
- Cryptocurrency markets are highly influenced by social sentiment, news, and technical factors
- Markets can exhibit high volatility and rapid directional changes
- Content credibility varies widely across different sources
- Regulatory news typically has outsized impact compared to other factors

Text to Analyze:
{text}

Instructions:
1. Analyze the text for explicit and implicit sentiment indicators.
2. Consider financial jargon and crypto-specific terminology.
3. Evaluate the market sentiment (bullish, bearish, or neutral).
4. Assess the credibility and potential impact of the content.
5. Identify key points that informed your analysis.
6. Provide a confidence level for your assessment.

Your response must be in the following JSON format:
{
    "sentiment_value": <float between 0 and 1, where 0 is extremely bearish, 0.5 is neutral, and 1 is extremely bullish>,
    "direction": <"bullish", "bearish", or "neutral">,
    "confidence": <float between 0 and 1 indicating your confidence level>,
    "explanation": <brief explanation of your reasoning>,
    "key_points": <list of key points that influenced your assessment>,
    "credibility_assessment": <assessment of the source's credibility and reliability>
}
"""
                },
                "control": False
            }
        ],
        targeting=[TargetingCriteria.ALL_TRAFFIC],
        assignment_strategy=VariantAssignmentStrategy.RANDOM,
        sample_size=20,  # Small sample size for demo
        min_confidence=0.95,
        owner="demo",
        metadata={"auto_generated": True}  # Mark as auto-generated for the CI system
    )
    
    # Start the experiment
    ab_testing_framework.start_experiment(experiment.id)
    
    logger.info(f"Created and started demo experiment: {experiment.id}")
    return experiment.id


async def check_experiment_status(experiment_id):
    """Check the status of an experiment.
    
    Args:
        experiment_id: ID of the experiment to check
    """
    experiment = ab_testing_framework.get_experiment(experiment_id)
    if not experiment:
        logger.error(f"Experiment {experiment_id} not found")
        return
    
    logger.info(f"Experiment Status: {experiment.status.value}")
    
    # Check variant metrics
    metrics_report = "\nVariant Metrics:\n"
    for variant in experiment.variants:
        metrics = experiment.variant_metrics[variant.id]
        metrics_report += f"- {variant.name}:\n"
        metrics_report += f"  Requests: {metrics.requests}\n"
        metrics_report += f"  Success Rate: {metrics.get_success_rate():.2f}\n"
        metrics_report += f"  Avg Latency: {metrics.get_average_latency():.2f} ms\n"
        metrics_report += f"  Sentiment Accuracy: {metrics.sentiment_accuracy:.3f}\n"
        metrics_report += f"  Direction Accuracy: {metrics.direction_accuracy:.3f}\n"
        metrics_report += f"  Confidence Score: {metrics.confidence_score:.3f}\n"
    
    logger.info(metrics_report)
    
    # Check if experiment has results
    if experiment.results:
        results_report = "\nExperiment Results:\n"
        results_report += f"Has significant results: {experiment.results.get('has_significant_results', False)}\n"
        results_report += f"Has clear winner: {experiment.results.get('has_clear_winner', False)}\n"
        results_report += f"Winning variant: {experiment.results.get('winning_variant')}\n"
        results_report += f"Recommendation: {experiment.results.get('recommendation')}\n"
        
        logger.info(results_report)


async def trigger_maintenance():
    """Trigger maintenance tasks in the continuous improvement system."""
    logger.info("Triggering continuous improvement maintenance...")
    
    await continuous_improvement_manager.run_maintenance()
    
    logger.info("Maintenance complete")


async def demonstrate_auto_improvement():
    """Demonstrate the auto-improvement capabilities."""
    logger.info("Demonstrating auto-improvement capabilities...")
    
    # Create a demo experiment with a predefined winner
    experiment_id = await generate_demo_experiment()
    experiment = ab_testing_framework.get_experiment(experiment_id)
    
    # Simulate several analysis requests to generate data
    llm_service = await setup_demo()
    await simulate_sentiment_analysis(llm_service, iterations=30)
    
    # Check the experiment status
    await check_experiment_status(experiment_id)
    
    # Simulate more requests to increase the sample size
    await simulate_sentiment_analysis(llm_service, iterations=20)
    
    # Check the experiment status again
    await check_experiment_status(experiment_id)
    
    # Complete the experiment
    logger.info("Completing the experiment for demonstration...")
    ab_testing_framework.complete_experiment(experiment_id)
    
    # Run an analysis on the experiment
    experiment.analyze_results()
    
    # Simulate a strong winner by manipulating the results
    logger.info("Simulating a clear winner for demonstration...")
    winning_variant = experiment.variants[1]  # Enhanced template
    experiment.results["has_clear_winner"] = True
    experiment.results["winning_variant"] = winning_variant.name
    experiment.results["recommendation"] = f"Implement variant '{winning_variant.name}' as it shows significant improvements."
    experiment.status = ExperimentStatus.ANALYZED
    
    # Check the experiment status with simulated results
    await check_experiment_status(experiment_id)
    
    # Trigger maintenance to auto-implement the winning variant
    logger.info("Triggering maintenance to auto-implement the winning variant...")
    await continuous_improvement_manager.run_maintenance()
    
    # Check for the implementation record
    improvement_history = continuous_improvement_manager.get_improvement_history()
    
    if improvement_history:
        logger.info("\nImprovement history:")
        for record in improvement_history:
            action = record.get("action", "")
            timestamp = record.get("timestamp", "")
            
            if action == "improvement_implemented":
                details = record.get("details", {})
                experiment_name = details.get("experiment_name", "")
                winning_variant = details.get("winning_variant", "")
                
                logger.info(f"- {timestamp}: Implemented {experiment_name} with winning variant {winning_variant}")
    else:
        logger.info("No improvements recorded yet")
    
    # Close the LLM service
    await llm_service.close()


async def generate_opportunities():
    """Demonstrate opportunity identification and experiment generation."""
    logger.info("Identifying improvement opportunities...")
    
    # Mock metrics for opportunity identification
    metrics = {
        "sentiment_accuracy": 0.82,
        "direction_accuracy": 0.75,
        "confidence_score": 0.65,
        "calibration_error": 0.12,
        "success_rate": 0.92,
        "average_latency": 450,
        "by_source": {
            "social_media": {"sentiment_accuracy": 0.75},
            "news": {"sentiment_accuracy": 0.85},
            "market_sentiment": {"sentiment_accuracy": 0.80},
            "onchain": {"sentiment_accuracy": 0.70}
        },
        "by_market_condition": {
            "bullish": {"sentiment_accuracy": 0.85},
            "bearish": {"sentiment_accuracy": 0.75},
            "neutral": {"sentiment_accuracy": 0.80},
            "volatile": {"sentiment_accuracy": 0.70}
        }
    }
    
    # Identify opportunities
    opportunities = continuous_improvement_manager._identify_improvement_opportunities(metrics)
    
    logger.info(f"Identified {len(opportunities)} improvement opportunities:")
    for i, opp in enumerate(opportunities):
        logger.info(f"{i+1}. {opp['type'].value}: {opp['reason']} (Impact: {opp['potential_impact']:.3f})")
    
    # Generate experiments (if enabled)
    if continuous_improvement_manager.enabled:
        logger.info("\nGenerating experiments from opportunities...")
        await continuous_improvement_manager.generate_experiments()
        
        # Check active experiments
        active_experiments = []
        for exp_id in ab_testing_framework.active_experiment_ids:
            exp = ab_testing_framework.get_experiment(exp_id)
            if exp and exp.metadata.get("auto_generated", False):
                active_experiments.append(exp)
        
        logger.info(f"Generated {len(active_experiments)} experiments:")
        for exp in active_experiments:
            logger.info(f"- {exp.name} ({exp.experiment_type.value})")
    else:
        logger.info("Continuous improvement system is disabled, skipping experiment generation")


async def main():
    """Main function to run the continuous improvement demo."""
    logger.info("Starting Continuous Improvement Demo")
    
    try:
        # Set up demo
        await setup_demo()
        
        # Show menu for demo options
        while True:
            print("\nContinuous Improvement Demo Menu:")
            print("1. Demonstrate Auto-Improvement")
            print("2. Generate Improvement Opportunities")
            print("3. Simulate Sentiment Analysis Requests")
            print("4. Trigger Maintenance Task")
            print("5. View System Status")
            print("6. Exit")
            
            choice = input("\nEnter your choice (1-6): ")
            
            if choice == '1':
                # Demonstrate auto-improvement
                await demonstrate_auto_improvement()
                
            elif choice == '2':
                # Generate opportunities
                await generate_opportunities()
                
            elif choice == '3':
                # Simulate sentiment analysis
                llm_service = await setup_demo()
                iterations = int(input("Enter number of iterations: ") or "20")
                await simulate_sentiment_analysis(llm_service, iterations)
                await llm_service.close()
                
            elif choice == '4':
                # Trigger maintenance
                await trigger_maintenance()
                
            elif choice == '5':
                # View system status
                status = continuous_improvement_manager.get_status()
                print("\nContinuous Improvement System Status:")
                print(f"Enabled: {status.get('enabled', False)}")
                print(f"Auto-implement: {status.get('auto_implement', False)}")
                print(f"Last check: {status.get('last_check', '')}")
                print(f"Last experiment generation: {status.get('last_experiment_generation', '')}")
                print(f"Active experiments: {status.get('active_experiments', 0)}")
                print(f"Improvements count: {status.get('improvements_count', 0)}")
                
            elif choice == '6':
                # Exit
                break
                
            else:
                print("Invalid choice. Please try again.")
                
    except Exception as e:
        logger.error(f"Error in demo: {e}")


if __name__ == "__main__":
    asyncio.run(main())