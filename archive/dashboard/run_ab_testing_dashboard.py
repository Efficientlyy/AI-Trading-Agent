#!/usr/bin/env python
"""
Standalone A/B Testing Dashboard for Sentiment Analysis.

This script runs the A/B testing dashboard as a standalone application.
"""

import asyncio
import logging
import argparse
import os
import sys
import json
import time
from datetime import datetime, timedelta

import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
from fastapi import FastAPI
from fastapi.middleware.wsgi import WSGIMiddleware

from src.analysis_agents.sentiment.ab_testing import ab_testing_framework, ExperimentType
from src.dashboard.components.ab_testing_dashboard import create_layout
from src.common.config import config


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/ab_testing_dashboard.log")
    ]
)

logger = logging.getLogger("ab_testing_dashboard")


async def initialize_dependencies():
    """Initialize required dependencies."""
    try:
        # Initialize the A/B testing framework
        await ab_testing_framework.initialize()
        
        # Create some demo experiments if none exist
        if not ab_testing_framework.experiments:
            logger.info("Creating demo experiments")
            create_demo_experiments()
        
        logger.info("Dependencies initialized successfully")
        
    except Exception as e:
        logger.error(f"Error initializing dependencies: {e}")
        raise


def create_demo_experiments():
    """Create demo experiments for testing."""
    # Demo experiment 1: Prompt template comparison
    template1 = """
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

    template2 = """
You are an elite financial sentiment analyzer with deep experience in cryptocurrency markets.
Your specialty is extracting sentiment signals from text about digital assets.

Text:
{text}

Instructions:
1. Analyze the text for explicit and implicit sentiment signals.
2. Evaluate the text's bullish, bearish, or neutral orientation toward crypto markets.
3. Consider both factual statements and emotional tone in your analysis.
4. Weigh the credibility and potential market impact of the information.
5. Identify specific catalysts or concerns mentioned in the text.

Provide your sentiment assessment in the following JSON format:
{
    "sentiment_value": <float from 0.0-1.0 where 0.0=extremely bearish, 0.5=neutral, 1.0=extremely bullish>,
    "direction": <"bullish", "bearish", or "neutral">,
    "confidence": <float from 0.0-1.0 indicating your assessment confidence>,
    "explanation": <concise explanation of key factors influencing your assessment>,
    "key_points": <list of 2-5 specific points from the text that determined your sentiment rating>
}
"""

    ab_testing_framework.create_experiment(
        name="Prompt Template Comparison",
        description="Comparing standard template vs. enhanced template for sentiment analysis",
        experiment_type=ExperimentType.PROMPT_TEMPLATE,
        variants=[
            {
                "name": "Standard Template",
                "description": "Default template for sentiment analysis",
                "weight": 0.5,
                "config": {"template": template1},
                "control": True
            },
            {
                "name": "Enhanced Template",
                "description": "Template with improved instructions and structure",
                "weight": 0.5,
                "config": {"template": template2},
                "control": False
            }
        ],
        sample_size=100,
        min_confidence=0.95
    )
    
    # Demo experiment 2: Model selection
    ab_testing_framework.create_experiment(
        name="Model Selection Test",
        description="Testing GPT-4o vs Claude-3 for sentiment analysis",
        experiment_type=ExperimentType.MODEL_SELECTION,
        variants=[
            {
                "name": "GPT-4o",
                "description": "Using GPT-4o as the LLM",
                "weight": 0.5,
                "config": {"model": "gpt-4o"},
                "control": True
            },
            {
                "name": "Claude-3",
                "description": "Using Claude-3 as the LLM",
                "weight": 0.5,
                "config": {"model": "claude-3-opus"},
                "control": False
            }
        ],
        sample_size=100,
        min_confidence=0.95
    )
    
    # Demo experiment 3: Temperature testing
    ab_testing_framework.create_experiment(
        name="Temperature Parameter Testing",
        description="Testing different temperature settings for response consistency",
        experiment_type=ExperimentType.TEMPERATURE,
        variants=[
            {
                "name": "Low Temperature",
                "description": "Temperature setting of 0.1 for more consistent results",
                "weight": 0.5,
                "config": {"temperature": 0.1},
                "control": True
            },
            {
                "name": "Medium Temperature",
                "description": "Temperature setting of 0.5 for balanced results",
                "weight": 0.5,
                "config": {"temperature": 0.5},
                "control": False
            }
        ],
        sample_size=100,
        min_confidence=0.95
    )


def create_app():
    """Create and configure the Dash app."""
    # Initialize the FastAPI app
    server = FastAPI(title="A/B Testing Dashboard")
    
    # Create Dash app
    app = dash.Dash(
        __name__,
        server=server,
        url_base_pathname="/",
        external_stylesheets=[dbc.themes.BOOTSTRAP]
    )
    
    # Set layout
    app.layout = html.Div([
        dbc.Container([
            html.Div([
                html.H1("A/B Testing Dashboard", className="text-primary mt-4 mb-3"),
                html.P("Monitor and manage A/B tests for the sentiment analysis system", className="lead mb-4"),
            ]),
            
            # Main dashboard content
            create_layout()
        ], fluid=True)
    ])
    
    # Mount Dash app on FastAPI
    server.mount("/", WSGIMiddleware(app.server))
    
    return server


async def maintenance_task():
    """Background task for A/B testing framework maintenance."""
    while True:
        try:
            await ab_testing_framework.run_maintenance()
            logger.info("A/B testing maintenance completed")
        except Exception as e:
            logger.error(f"Error in A/B testing maintenance: {e}")
        
        # Sleep for 6 hours
        await asyncio.sleep(21600)


async def start_maintenance_task():
    """Start the background maintenance task."""
    asyncio.create_task(maintenance_task())


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run the A/B Testing Dashboard")
    parser.add_argument(
        "--host",
        type=str,
        default=config.get("sentiment_analysis.ab_testing.dashboard.host", "0.0.0.0"),
        help="Host to run the server on"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=config.get("sentiment_analysis.ab_testing.dashboard.port", 8052),
        help="Port to run the server on"
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        default=False,
        help="Enable auto-reloading"
    )
    return parser.parse_args()


async def main():
    """Main function."""
    # Parse arguments
    args = parse_arguments()
    
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    # Initialize dependencies
    await initialize_dependencies()
    
    # Start maintenance task
    await start_maintenance_task()
    
    # Create app
    app = create_app()
    
    # Import uvicorn here to avoid any startup conflicts
    import uvicorn
    
    # Show server info
    logger.info(f"Starting A/B Testing Dashboard on {args.host}:{args.port}")
    
    # Run the server
    config = uvicorn.Config(
        app=app,
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    )
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())