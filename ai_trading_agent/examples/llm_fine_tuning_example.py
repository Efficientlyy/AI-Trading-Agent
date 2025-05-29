#!/usr/bin/env python3
"""
Example demonstrating the LLM fine-tuning functionality of the AI Trading Agent.

This script shows how to:
1. Configure the fine-tuning process
2. Prepare financial domain datasets
3. Fine-tune a language model
4. Evaluate the fine-tuned model
5. Use the fine-tuned model for financial analysis

Usage:
    python llm_fine_tuning_example.py
"""
import os
import sys
import logging
from pathlib import Path
import json
from datetime import datetime

# Add the parent directory to the path to import the ai_trading_agent package
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from ai_trading_agent.llm.fine_tuning import (
    FineTuningConfig,
    FinancialDatasetPreparer,
    ModelFineTuner,
    ModelEvaluator,
    FinancialPromptTemplate,
    get_prompt_template
)
from ai_trading_agent.llm.fine_tuning.config import (
    ModelProvider,
    FineTuningMethod,
    DatasetType
)
from ai_trading_agent.llm.fine_tuning.prompt_templates import PromptTaskType


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_fine_tuning_config():
    """Set up the fine-tuning configuration."""
    # Create a configuration for financial domain fine-tuning
    config = FineTuningConfig(
        # Model configuration
        model_provider=ModelProvider.HUGGING_FACE,  # Use Hugging Face for local fine-tuning
        base_model="mistralai/Mistral-7B-v0.1",  # Use Mistral 7B as base model
        fine_tuning_method=FineTuningMethod.LORA,  # Use LoRA for parameter-efficient fine-tuning
        
        # Dataset configuration
        dataset_types=[DatasetType.FINANCIAL_NEWS, DatasetType.MARKET_COMMENTARY],
        dataset_paths={
            DatasetType.FINANCIAL_NEWS: "./data/financial_news",
            DatasetType.MARKET_COMMENTARY: "./data/market_commentary"
        },
        validation_split=0.1,
        test_split=0.1,
        
        # Training parameters
        batch_size=4,
        num_epochs=3,
        learning_rate=2e-5,
        
        # LoRA parameters
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        
        # Output configuration
        output_dir="./fine_tuned_models/financial_domain"
    )
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Save configuration
    config.save(os.path.join(config.output_dir, "config.json"))
    logger.info(f"Fine-tuning configuration saved to {os.path.join(config.output_dir, 'config.json')}")
    
    return config


def create_sample_dataset():
    """Create a sample dataset for demonstration purposes."""
    # Create dataset directories
    news_dir = "./data/financial_news"
    commentary_dir = "./data/market_commentary"
    os.makedirs(news_dir, exist_ok=True)
    os.makedirs(commentary_dir, exist_ok=True)
    
    # Sample financial news data
    financial_news = [
        {
            "headline": "Tech Stocks Rally Despite Economic Concerns",
            "text": "The tech sector showed remarkable resilience today, with major tech stocks rallying despite broader economic concerns. Analysts point to strong earnings reports and positive forward guidance as key factors supporting investor confidence in the technology space. The NASDAQ composite gained 2.3%, outperforming other major indices.",
            "date": "2025-05-15",
            "sector": "Technology",
            "sentiment": "positive"
        },
        {
            "headline": "Federal Reserve Signals Potential Rate Cut",
            "text": "In a surprising shift of tone, Federal Reserve officials today signaled openness to potential interest rate cuts later this year. The announcement came after inflation data showed continued moderation in price pressures across multiple sectors. Bond markets rallied on the news, with yields dropping sharply.",
            "date": "2025-05-14",
            "sector": "Financials",
            "sentiment": "positive"
        },
        {
            "headline": "Oil Prices Surge Amid Supply Concerns",
            "text": "Crude oil prices surged more than 4% today following reports of production disruptions in key oil-producing regions. The supply constraints come at a time of increasing global demand, creating a tight market that analysts expect could push prices even higher in the coming weeks. Energy stocks benefited from the price movement.",
            "date": "2025-05-13",
            "sector": "Energy",
            "sentiment": "mixed"
        }
    ]
    
    # Sample market commentary data
    market_commentary = [
        {
            "title": "Weekly Market Outlook: Navigating Volatility",
            "text": "This week presents a challenging environment for investors as market volatility remains elevated. The VIX index, often referred to as the 'fear gauge,' continues to hover above historical averages, suggesting sustained uncertainty. Despite this backdrop, selective opportunities exist in defensive sectors and quality growth names with strong balance sheets. We recommend a balanced approach with strategic hedging to navigate the current market conditions.",
            "date": "2025-05-16",
            "author": "Investment Strategy Team",
            "focus": "Market Strategy"
        },
        {
            "title": "Sector Rotation: Where to Position Now",
            "text": "We're observing a notable rotation from growth to value stocks as interest rate expectations shift. Financials, particularly regional banks with strong deposit bases, look attractive under this scenario. Additionally, healthcare stocks with limited economic sensitivity offer both defensive characteristics and reasonable valuations. Conversely, we're becoming more cautious on consumer discretionary exposure given potential pressure on household spending in the coming quarters.",
            "date": "2025-05-15",
            "author": "Sector Analysis Team",
            "focus": "Sector Strategy"
        },
        {
            "title": "Technical Analysis Alert: S&P 500 Approaching Key Resistance",
            "text": "The S&P 500 is approaching a critical technical resistance level at 5,850, which has rejected advances twice in recent months. Volume patterns suggest accumulation is occurring below this threshold, potentially providing the momentum needed for a breakout. If successful, the next target would be the 6,000 psychological level. However, failure to break through could lead to a retest of support around 5,650. Traders should watch for expanding volume on any decisive move in either direction.",
            "date": "2025-05-14",
            "author": "Technical Analysis Team",
            "focus": "Technical Signals"
        }
    ]
    
    # Save sample data
    with open(os.path.join(news_dir, "financial_news.json"), "w") as f:
        json.dump(financial_news, f, indent=2)
    
    with open(os.path.join(commentary_dir, "market_commentary.json"), "w") as f:
        json.dump(market_commentary, f, indent=2)
    
    logger.info(f"Created sample datasets in {news_dir} and {commentary_dir}")


def prepare_financial_datasets(config):
    """Prepare financial datasets for fine-tuning."""
    logger.info("Preparing financial datasets for fine-tuning")
    
    # Create dataset preparer
    dataset_preparer = FinancialDatasetPreparer(config)
    
    # Collect datasets
    datasets = dataset_preparer.collect_datasets()
    logger.info(f"Collected {len(datasets)} datasets")
    
    # Prepare data for fine-tuning
    prepared_data = dataset_preparer.prepare_data()
    logger.info(f"Prepared data splits: {', '.join([f'{k}: {len(v)}' for k, v in prepared_data.items()])}")
    
    return prepared_data


def fine_tune_model(config, prepared_data):
    """Fine-tune the language model on financial data."""
    logger.info("Starting language model fine-tuning")
    
    # Create model fine-tuner
    model_fine_tuner = ModelFineTuner(config)
    
    # Fine-tune the model
    try:
        # For demonstration purposes, we'll just print what would happen
        # In a real scenario, this would actually fine-tune the model
        logger.info("In a real scenario, fine-tuning would start now with the following configuration:")
        logger.info(f"  Model provider: {config.model_provider.value}")
        logger.info(f"  Base model: {config.base_model}")
        logger.info(f"  Fine-tuning method: {config.fine_tuning_method.value}")
        logger.info(f"  Number of training examples: {len(prepared_data['train'])}")
        
        # To avoid the actual computation-intensive process, we'll just simulate the process
        logger.info("Simulating fine-tuning process (this would take hours/days in reality)")
        
        # In a real scenario, you would uncomment this line to actually fine-tune
        # model_fine_tuner.fine_tune(prepared_data)
        
        logger.info("Fine-tuning simulation completed")
        
        # Create a mock fine-tuning info file for demonstration
        fine_tuning_info = {
            "provider": config.model_provider.value,
            "base_model": config.base_model,
            "method": config.fine_tuning_method.value,
            "completed_at": datetime.now().isoformat(),
            "train_metrics": {"loss": 0.4325, "perplexity": 10.23},
            "eval_metrics": {"loss": 0.4892, "perplexity": 12.15}
        }
        
        with open(os.path.join(config.output_dir, "fine_tuning_info.json"), "w") as f:
            json.dump(fine_tuning_info, f, indent=2)
            
        return True
        
    except Exception as e:
        logger.error(f"Error during fine-tuning: {e}")
        return False


def evaluate_fine_tuned_model(config, test_data):
    """Evaluate the fine-tuned model performance."""
    logger.info("Evaluating fine-tuned model")
    
    # Create model evaluator
    evaluator = ModelEvaluator(config)
    
    # For demonstration, we'll simulate the evaluation
    logger.info("In a real scenario, model evaluation would run now")
    logger.info(f"Number of test examples: {len(test_data)}")
    
    # Simulate evaluation with mock results
    mock_metrics = {
        "financial_accuracy": 0.87,
        "regulatory_compliance": 0.92,
        "reasoning_quality": 0.85,
        "factual_consistency": 0.83,
        "domain_specificity": 0.91,
        "overall_score": 0.88
    }
    
    # Save mock evaluation results
    evaluation_results = {
        "model_provider": config.model_provider.value,
        "model_id": os.path.join(config.output_dir, "final_model"),
        "metrics": mock_metrics,
        "example_results": [
            {"prompt": "Example prompt", "response": "Example response", "metrics": mock_metrics}
        ],
        "timestamp": datetime.now().isoformat()
    }
    
    with open(os.path.join(config.output_dir, "evaluation_results.json"), "w") as f:
        json.dump(evaluation_results, f, indent=2)
    
    logger.info("Evaluation simulation completed")
    logger.info(f"Overall score: {mock_metrics['overall_score']:.4f}")
    
    return mock_metrics


def demonstrate_prompt_templates():
    """Demonstrate the use of financial prompt templates."""
    logger.info("Demonstrating financial prompt templates")
    
    # List available task types
    task_types = FinancialPromptTemplate.list_available_tasks()
    logger.info(f"Available prompt task types: {', '.join(task_types)}")
    
    # Create a market analysis prompt template
    market_analysis_template = get_prompt_template("market_analysis")
    
    # Format the template with sample data
    formatted_prompt = market_analysis_template.format(
        market="U.S. Equity Markets - S&P 500",
        timeframe="Short-term (1-2 weeks)",
        indicators="Moving averages, RSI, MACD, Volume trends",
        events="Recent Fed meeting, Quarterly earnings season",
        prior_analysis="Previously identified resistance at 5,850 level",
        context="The S&P 500 has been trending upward for three consecutive weeks, approaching the 5,850 resistance level. Volume has been increasing during up days, suggesting accumulation. The RSI is at 68, approaching but not yet in overbought territory. The MACD shows a bullish crossover pattern. Recent earnings reports have been mixed, with technology and financial sectors outperforming expectations while consumer discretionary has lagged."
    )
    
    # Print the formatted prompt
    logger.info("Example Market Analysis Prompt:")
    logger.info(f"System prompt: {formatted_prompt['system_prompt'][:100]}...")
    logger.info(f"User prompt: {formatted_prompt['user_prompt'][:100]}...")
    
    # Save example prompts
    example_prompts_dir = os.path.join("./fine_tuned_models/financial_domain", "example_prompts")
    os.makedirs(example_prompts_dir, exist_ok=True)
    
    # Create and save prompts for different tasks
    for task_type in task_types:
        template = get_prompt_template(task_type)
        
        # Create a sample formatted prompt based on task type
        if task_type == "market_analysis":
            formatted = template.format(
                market="U.S. Equity Markets - S&P 500",
                timeframe="Short-term (1-2 weeks)",
                indicators="Moving averages, RSI, MACD, Volume trends",
                events="Recent Fed meeting, Quarterly earnings season",
                context="Market has been trending upward with increasing volume."
            )
        elif task_type == "trading_strategy":
            formatted = template.format(
                market="EUR/USD Forex",
                timeframe="Day trading (intraday)",
                risk_tolerance="Moderate",
                capital="$100,000",
                constraints="No overnight positions",
                context="EUR/USD has been range-bound between 1.05 and 1.08 for the past month."
            )
        else:
            # Generic formatting for other task types
            formatted = template.format(
                context="This is a sample context for demonstrating the prompt template.",
                **{field: "Sample data" for field in template.required_fields if field != "context"}
            )
        
        # Save this example
        with open(os.path.join(example_prompts_dir, f"{task_type}_example.json"), "w") as f:
            json.dump(formatted, f, indent=2)
    
    logger.info(f"Saved example prompts to {example_prompts_dir}")


def demonstrate_inference(config):
    """Demonstrate inference with a fine-tuned model."""
    logger.info("Demonstrating inference with fine-tuned model")
    
    # In a real scenario, you would load the fine-tuned model
    # model_fine_tuner = ModelFineTuner(config)
    # model_fine_tuner.load_fine_tuned_model()
    
    # For demonstration, we'll simulate model responses
    sample_prompts = [
        "Analyze the current market conditions for U.S. equities considering recent inflation data and Fed policy statements.",
        "Develop a trading strategy for gold futures in the current market environment with a moderate risk tolerance.",
        "Assess the regulatory compliance implications of implementing a new algorithmic trading strategy under current SEC guidelines."
    ]
    
    logger.info("Sample inferences with fine-tuned model:")
    for i, prompt in enumerate(sample_prompts):
        # Simulate model response
        simulated_response = f"This is a simulated response for prompt {i+1}. In a real scenario, this would be generated by the fine-tuned model with financial domain expertise."
        
        logger.info(f"Prompt {i+1}: {prompt[:50]}...")
        logger.info(f"Response: {simulated_response[:50]}...")
    
    logger.info("Inference demonstration completed")


def main():
    """Main function to demonstrate LLM fine-tuning workflow."""
    logger.info("Starting LLM fine-tuning example")
    
    # Set up configuration
    config = setup_fine_tuning_config()
    
    # Create sample datasets
    create_sample_dataset()
    
    # Prepare datasets
    prepared_data = prepare_financial_datasets(config)
    
    # Fine-tune model (simulated)
    fine_tuning_success = fine_tune_model(config, prepared_data)
    
    if fine_tuning_success:
        # Evaluate model (simulated)
        evaluation_metrics = evaluate_fine_tuned_model(config, prepared_data["test"])
        
        # Demonstrate prompt templates
        demonstrate_prompt_templates()
        
        # Demonstrate inference (simulated)
        demonstrate_inference(config)
    
    logger.info("LLM fine-tuning example completed")


if __name__ == "__main__":
    main()
