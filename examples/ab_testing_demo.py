#!/usr/bin/env python
"""
A/B Testing Demo for Sentiment Analysis.

This script demonstrates how to create and manage A/B tests for the sentiment
analysis system, shows how to create experiments, monitor results, and implement
winners.
"""

import asyncio
import logging
import json
import time
import random
from datetime import datetime, timedelta

from src.analysis_agents.sentiment.llm_service import LLMService
from src.analysis_agents.sentiment.ab_testing import (
    ab_testing_framework, ExperimentType, TargetingCriteria, 
    VariantAssignmentStrategy, ExperimentStatus
)
from src.common.logging import get_logger


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = get_logger("examples", "ab_testing_demo")


async def setup_demo():
    """Set up the demo by initializing required services."""
    # Initialize the LLM service (which also initializes the A/B testing framework)
    llm_service = LLMService()
    await llm_service.initialize()
    
    # Ensure A/B testing framework is initialized
    await ab_testing_framework.initialize()
    
    return llm_service


async def create_test_experiment():
    """Create a test experiment for the demo."""
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
        owner="demo"
    )
    
    logger.info(f"Created experiment: {experiment.id}")
    return experiment


async def run_experiment_demo(llm_service, experiment_id):
    """Run a demo of the experiment.
    
    Args:
        llm_service: LLM service instance
        experiment_id: ID of the experiment to run
    """
    # Get the experiment
    experiment = ab_testing_framework.get_experiment(experiment_id)
    if not experiment:
        logger.error(f"Experiment {experiment_id} not found")
        return
    
    # Start the experiment
    if not ab_testing_framework.start_experiment(experiment_id):
        logger.error(f"Failed to start experiment {experiment_id}")
        return
    
    logger.info(f"Started experiment: {experiment.name}")
    
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
    
    # Analyze each text multiple times to get variant distribution
    iterations = 3  # Run each text 3 times for demo purposes
    
    logger.info(f"Analyzing {len(sample_texts)} texts with {iterations} iterations each...")
    
    for i in range(iterations):
        for text in sample_texts:
            # Create a random context ID to simulate different requests
            context_id = f"ctx_{random.randint(1000, 9999)}"
            symbol = random.choice(["BTC", "ETH", "SOL", "XRP"])
            
            logger.info(f"Analysis iteration {i+1} for context {context_id}")
            
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
    
    # Check experiment status
    logger.info("Experiment run completed. Checking experiment status...")
    experiment = ab_testing_framework.get_experiment(experiment_id)
    
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


async def demonstrate_experiment_lifecycle():
    """Demonstrate the full lifecycle of an experiment."""
    # Create the experiment
    experiment = await create_test_experiment()
    
    # Show experiment details before starting
    logger.info(f"Experiment created: {experiment.name}")
    logger.info(f"Status: {experiment.status.value}")
    logger.info(f"Type: {experiment.experiment_type.value}")
    logger.info(f"Variants: {len(experiment.variants)}")
    
    # Start the experiment
    ab_testing_framework.start_experiment(experiment.id)
    logger.info(f"Experiment started: {experiment.name}")
    
    # Simulate some data
    for i in range(5):
        for variant in experiment.variants:
            # Simulate success and latency
            success = random.random() > 0.1  # 90% success rate
            latency = random.uniform(100, 500)  # 100-500ms latency
            
            # Create a context
            context = {
                "symbol": random.choice(["BTC", "ETH", "SOL"]),
                "source": random.choice(["news", "social_media", "market_data"]),
                "request_id": f"req_{i}_{variant.id}"
            }
            
            # Record a result
            result = {
                "success": success,
                "latency_ms": latency,
                "sentiment_accuracy": random.uniform(0.7, 0.95) if success else 0,
                "direction_accuracy": random.uniform(0.7, 0.95) if success else 0,
                "confidence_score": random.uniform(0.6, 0.9) if success else 0
            }
            
            experiment.record_result(variant.id, result, context)
    
    # Sleep a bit to simulate time passing
    logger.info("Simulating experiment running...")
    await asyncio.sleep(2)
    
    # Now complete the experiment
    ab_testing_framework.complete_experiment(experiment.id)
    logger.info(f"Experiment completed: {experiment.name}")
    
    # Analyze results
    experiment.analyze_results()
    logger.info(f"Experiment analysis complete")
    
    # Show analysis results
    analysis = experiment.results
    logger.info(f"Analysis results:")
    logger.info(f"- Has significant results: {analysis.get('has_significant_results', False)}")
    logger.info(f"- Has clear winner: {analysis.get('has_clear_winner', False)}")
    logger.info(f"- Winning variant: {analysis.get('winning_variant')}")
    logger.info(f"- Recommendation: {analysis.get('recommendation')}")
    
    # Complete the lifecycle
    ab_testing_framework.archive_experiment(experiment.id)
    logger.info(f"Experiment archived: {experiment.name}")
    
    # Show final status
    status = experiment.get_status()
    logger.info(f"Final status: {status['status']}")


async def main():
    """Main function to run the A/B testing demo."""
    logger.info("Starting A/B Testing Demo")
    
    try:
        # Set up demo
        llm_service = await setup_demo()
        
        # Show menu for demo options
        while True:
            print("\nA/B Testing Demo Menu:")
            print("1. Create and run a test experiment")
            print("2. List all experiments")
            print("3. Demonstrate experiment lifecycle")
            print("4. View experiment details")
            print("5. Start/pause/complete an experiment")
            print("6. Exit")
            
            choice = input("\nEnter your choice (1-6): ")
            
            if choice == '1':
                # Create and run experiment
                experiment = await create_test_experiment()
                await run_experiment_demo(llm_service, experiment.id)
                
            elif choice == '2':
                # List experiments
                experiments = ab_testing_framework.list_experiments()
                if not experiments:
                    print("No experiments found")
                else:
                    print(f"\nFound {len(experiments)} experiments:")
                    for i, exp in enumerate(experiments):
                        print(f"{i+1}. {exp['name']} - Status: {exp['status']}, Type: {exp['type']}")
                
            elif choice == '3':
                # Demonstrate lifecycle
                await demonstrate_experiment_lifecycle()
                
            elif choice == '4':
                # View experiment details
                experiments = ab_testing_framework.list_experiments()
                if not experiments:
                    print("No experiments found")
                    continue
                    
                print("\nAvailable experiments:")
                for i, exp in enumerate(experiments):
                    print(f"{i+1}. {exp['name']} - Status: {exp['status']}")
                
                idx = input("\nEnter experiment number to view: ")
                try:
                    idx = int(idx) - 1
                    if 0 <= idx < len(experiments):
                        exp_id = experiments[idx]['id']
                        report = ab_testing_framework.create_experiment_report(exp_id)
                        print(f"\nExperiment: {report['name']}")
                        print(f"Status: {report['status']}")
                        print(f"Type: {report['type']}")
                        print(f"Created: {report['created_at']}")
                        print(f"Total Traffic: {report['total_traffic']}")
                        
                        if report.get('results'):
                            print(f"\nResults:")
                            print(f"Has significant results: {report['results'].get('has_significant_results', False)}")
                            print(f"Has clear winner: {report['results'].get('has_clear_winner', False)}")
                            print(f"Winning variant: {report['results'].get('winning_variant')}")
                            print(f"Recommendation: {report['recommendation']}")
                    else:
                        print("Invalid experiment number")
                except ValueError:
                    print("Invalid input")
                
            elif choice == '5':
                # Manage experiment status
                experiments = ab_testing_framework.list_experiments()
                if not experiments:
                    print("No experiments found")
                    continue
                    
                print("\nAvailable experiments:")
                for i, exp in enumerate(experiments):
                    print(f"{i+1}. {exp['name']} - Status: {exp['status']}")
                
                idx = input("\nEnter experiment number to manage: ")
                try:
                    idx = int(idx) - 1
                    if 0 <= idx < len(experiments):
                        exp_id = experiments[idx]['id']
                        exp = ab_testing_framework.get_experiment(exp_id)
                        
                        print(f"\nManage experiment: {exp.name}")
                        print(f"Current status: {exp.status.value}")
                        print("\nActions:")
                        print("1. Start experiment")
                        print("2. Pause experiment")
                        print("3. Complete experiment")
                        print("4. Analyze results")
                        print("5. Archive experiment")
                        
                        action = input("\nSelect action (1-5): ")
                        
                        if action == '1':
                            if ab_testing_framework.start_experiment(exp_id):
                                print("Experiment started successfully")
                            else:
                                print("Failed to start experiment")
                                
                        elif action == '2':
                            if ab_testing_framework.pause_experiment(exp_id):
                                print("Experiment paused successfully")
                            else:
                                print("Failed to pause experiment")
                                
                        elif action == '3':
                            if ab_testing_framework.complete_experiment(exp_id):
                                print("Experiment completed successfully")
                            else:
                                print("Failed to complete experiment")
                                
                        elif action == '4':
                            exp.analyze_results()
                            print("Experiment analysis completed")
                            
                        elif action == '5':
                            if ab_testing_framework.archive_experiment(exp_id):
                                print("Experiment archived successfully")
                            else:
                                print("Failed to archive experiment")
                        
                        else:
                            print("Invalid action")
                    else:
                        print("Invalid experiment number")
                except ValueError:
                    print("Invalid input")
                
            elif choice == '6':
                # Exit
                break
                
            else:
                print("Invalid choice. Please try again.")
                
    except Exception as e:
        logger.error(f"Error in demo: {e}")
    finally:
        # Clean up
        await llm_service.close()
        logger.info("A/B Testing Demo completed")


if __name__ == "__main__":
    asyncio.run(main())