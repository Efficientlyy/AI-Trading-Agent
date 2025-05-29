"""
AI Trading Agent API Server

This script runs a FastAPI server that provides API endpoints for the AI Trading Agent.
It exposes sentiment data from Alpha Vantage and other trading system components.

Usage:
    python api_server.py [--port PORT] [--host HOST] [--use-mock]
"""

import argparse
import logging
import os
import sys
import uvicorn
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Import necessary components
from ai_trading_agent.common.logging_config import setup_logging
from ai_trading_agent.api import api_router, websocket_router
from ai_trading_agent.agent import (
    TradingOrchestrator,
    SentimentAnalysisAgent,
    TechnicalAnalysisAgent,
    DecisionAgent,
    ExecutionLayerAgent,
    AgentStatus # For type hinting if needed
)

# Set up logging
setup_logging()
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def create_app():
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="AI Trading Agent API",
        description="API for AI Trading Agent",
        version="0.1.0"
    )
    
    # Initialize Orchestrator and Agents
    orchestrator = TradingOrchestrator()

    sentiment_agent = SentimentAnalysisAgent(
        agent_id_suffix="alphavantage_news_btc", name="AlphaVantage BTC News Sentiment",
        agent_type="AlphaVantageNews", symbols=["BTC/USD"]
    )
    technical_agent = TechnicalAnalysisAgent(
        agent_id_suffix="rsi_macd_eth", name="RSI/MACD ETH Technicals",
        agent_type="RSIMACDStrategy", symbols=["ETH/USD"]
    )
    decision_agent = DecisionAgent(
        agent_id_suffix="main_crypto_v1", name="Main Crypto Decision Logic V1",
        agent_type="WeightedSignalAggregator"
    )
    execution_agent = ExecutionLayerAgent(
        agent_id_suffix="alpaca_paper", name="Alpaca Paper Trading Executor",
        agent_type="AlpacaBroker"
    )

    # Define connections
    sentiment_agent.outputs_to = [decision_agent.agent_id]
    technical_agent.outputs_to = [decision_agent.agent_id]
    decision_agent.inputs_from = [sentiment_agent.agent_id, technical_agent.agent_id]
    decision_agent.outputs_to = [execution_agent.agent_id]
    execution_agent.inputs_from = [decision_agent.agent_id]

    # Register agents
    orchestrator.register_agent(sentiment_agent)
    orchestrator.register_agent(technical_agent)
    orchestrator.register_agent(decision_agent)
    orchestrator.register_agent(execution_agent)
    
    orchestrator.start_all_agents() # Start agents after registration and order determination

    app.state.orchestrator = orchestrator
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Adjust this in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include API router
    app.include_router(api_router) # This router is typically for /api prefixed routes
    
    # Include WebSocket router (mounted at root level, not under /api)
    app.include_router(websocket_router)

    # Dependency to get the orchestrator
    def get_orchestrator() -> TradingOrchestrator:
        return app.state.orchestrator

    @app.get("/api/agents/status")
    async def get_agents_status(orch: TradingOrchestrator = Depends(get_orchestrator)):
        return orch.get_all_agent_info()

    @app.post("/api/agents/run_cycle")
    async def run_agent_cycle(orch: TradingOrchestrator = Depends(get_orchestrator)):
        # This is a simplified trigger; in a real system, this might be timed or event-driven
        orch.run_cycle()
        return {"message": "Orchestrator cycle triggered."}
    
    @app.get("/")
    async def root():
        """Root endpoint."""
        return {
            "message": "AI Trading Agent API",
            "documentation": "/docs",
            "endpoints": [
                "/api/sentiment/summary",
                "/api/sentiment/historical"
            ]
        }
    
    return app

def main():
    """Run the API server."""
    try:
        # Parse command line arguments
        parser = argparse.ArgumentParser(description="Run the AI Trading Agent API server")
        parser.add_argument("--port", type=int, default=8000, help="Port to run the server on")
        parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to run the server on")
        parser.add_argument("--use-mock", action="store_true", help="Use mock data instead of real API calls")
        args = parser.parse_args()
        
        # Check for Alpha Vantage API key if not using mock data
        if not args.use_mock and not os.getenv("ALPHA_VANTAGE_API_KEY"):
            logger.warning("Alpha Vantage API key not found in environment variables")
            logger.warning("Set ALPHA_VANTAGE_API_KEY or use --use-mock to use mock data")
        
        # Create and run the application
        app = create_app()
        
        logger.info(f"Starting server on {args.host}:{args.port}")
        logger.info(f"Using {'mock' if args.use_mock else 'real'} data")
        logger.info(f"View API documentation at http://{args.host}:{args.port}/docs")
        
        uvicorn.run(app, host=args.host, port=args.port)
        
        return 0
    except Exception as e:
        logger.error(f"Error starting server: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main())