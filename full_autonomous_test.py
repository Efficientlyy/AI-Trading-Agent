#!/usr/bin/env python
"""
Comprehensive Autonomous System Test with Mock Data

This script tests the full AI Trading Agent system in autonomous mode using mock data sources
instead of requiring external API keys. It demonstrates the integration of all agents:
- Adaptive Health Orchestrator
- Market Regime Classification
- Sentiment Analysis
- Technical Analysis
- Risk Management
- LLM Oversight (if enabled)
"""

import os
import sys
import logging
import time
import random
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[
        logging.FileHandler("autonomous_test.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Import necessary components
try:
    from ai_trading_agent.agent.adaptive_orchestrator import AdaptiveHealthOrchestrator
    from ai_trading_agent.market_regime import MarketRegimeConfig
    from ai_trading_agent.agent.agent_definitions import BaseAgent, AgentRole, AgentStatus
    from ai_trading_agent.agent.strategy import SentimentStrategy
    from ai_trading_agent.strategies.ma_crossover_strategy import MACrossoverStrategy
    from ai_trading_agent.data_acquisition.mock_provider import MockDataProvider
    from ai_trading_agent.sentiment_analysis.mock_provider import MockSentimentProvider
    from ai_trading_agent.agent.meta_strategy import DynamicAggregationMetaStrategy
    from ai_trading_agent.common.logging_config import setup_logging
    
    # Set up proper logging
    setup_logging(log_level="INFO")
    
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    import traceback
    logger.error(f"Detailed error: {traceback.format_exc()}")
    sys.exit(1)

class MockLLMOversightClient:
    """Mock LLM Oversight Client for demonstration purposes."""
    
    def __init__(self):
        self.service_available = True
        logger.info("Initialized Mock LLM Oversight Client")
    
    def check_health(self):
        """Check if the LLM oversight service is available."""
        return self.service_available
    
    def get_config(self):
        """Get the configuration for the LLM oversight service."""
        return {
            "model": "mock-llm-model",
            "max_tokens": 1000,
            "temperature": 0.7,
            "oversight_level": "advise"
        }
    
    def validate_trading_decision(self, decision_data):
        """Validate a trading decision using the LLM oversight service."""
        # Randomly approve or suggest modifications
        action = random.choice(["approve", "modify", "log"])
        reason = None
        
        if action == "approve":
            reason = "Decision aligns with current market conditions and risk parameters."
        elif action == "modify":
            reason = "Suggested position size reduction due to increased market volatility."
            # Include modification suggestions
            return {
                "action": action,
                "reason": reason,
                "modified_decision": {
                    "position_size": decision_data.get("position_size", 1.0) * 0.8,
                    "stop_loss_pct": decision_data.get("stop_loss_pct", 0.05) * 1.25
                }
            }
        else:  # log
            reason = "Decision logged for monitoring purposes."
        
        return {
            "action": action,
            "reason": reason,
            "confidence": 0.85
        }
    
    def analyze_market_conditions(self, market_data):
        """Analyze market conditions using the LLM oversight service."""
        # Generate a mock market analysis
        regimes = ["bull", "bear", "sideways", "volatile"]
        regime = random.choice(regimes)
        
        return {
            "regime": regime,
            "analysis": f"Market shows signs of {regime} conditions with {random.randint(60, 90)}% confidence.",
            "outlook": random.choice(["bullish", "bearish", "neutral"]),
            "risk_assessment": random.choice(["low", "moderate", "high"]),
            "confidence": random.random() * 0.3 + 0.7  # 0.7-1.0
        }

def create_mock_data(symbols, days=30):
    """Create mock market data for testing."""
    logger.info(f"Generating mock market data for {symbols} over {days} days")
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    price_data = {}
    sentiment_data = {}
    
    for symbol in symbols:
        # Generate price data with realistic patterns
        seed_price = 100 if 'BTC' in symbol else (50 if 'ETH' in symbol else random.uniform(10, 200))
        price_series = [seed_price]
        
        # Generate price with random walk and some trend
        trend = random.choice([-0.01, 0.01])  # Small daily trend
        for _ in range(len(dates) - 1):
            daily_return = trend + random.normalvariate(0, 0.02)  # Random daily fluctuation
            new_price = price_series[-1] * (1 + daily_return)
            price_series.append(new_price)
        
        # Create price DataFrame
        df = pd.DataFrame({
            'timestamp': dates,
            'open': price_series,
            'high': [p * (1 + random.uniform(0, 0.02)) for p in price_series],
            'low': [p * (1 - random.uniform(0, 0.02)) for p in price_series],
            'close': price_series,
            'volume': [random.randint(100000, 1000000) for _ in range(len(dates))]
        })
        price_data[symbol] = df
        
        # Generate sentiment data with some correlation to price changes
        sentiment_values = []
        for i in range(len(dates)):
            # Some correlation with price trend
            if i > 0:
                price_change = (price_series[i] / price_series[i-1]) - 1
                # Sentiment has some correlation with price changes but also noise
                sent_value = 0.3 * np.sign(price_change) + random.uniform(-0.7, 0.7)
                # Clamp to [-1, 1]
                sent_value = max(min(sent_value, 1.0), -1.0)
            else:
                sent_value = random.uniform(-0.5, 0.5)
            
            sentiment_values.append(sent_value)
        
        sentiment_df = pd.DataFrame({
            'timestamp': dates,
            'symbol': symbol,
            'sentiment_score': sentiment_values,
            'source': 'mock',
            'volume': [random.randint(100, 1000) for _ in range(len(dates))]
        })
        sentiment_data[symbol] = sentiment_df
    
    return price_data, sentiment_data

def run_autonomous_system_test(symbols=['BTC/USD', 'ETH/USD'], test_duration_seconds=60):
    """
    Run a complete test of the autonomous trading system using mock data.
    
    Args:
        symbols: List of symbols to test with
        test_duration_seconds: How long to run the test in seconds
    """
    logger.info(f"Starting autonomous system test with symbols: {symbols}")
    
    # Create mock data
    price_data, sentiment_data = create_mock_data(symbols, days=60)
    
    # Initialize mock data providers with configuration
    mock_data_provider = MockDataProvider({
        "symbols": symbols,
        "timeframes": ["1d"],
        "use_default_patterns": True  # Generate realistic market patterns
    })
    
    # Create predefined sentiment data for the MockSentimentProvider
    sentiment_data_dict = {}
    for symbol, df in sentiment_data.items():
        sentiment_df = pd.DataFrame({
            'sentiment_score': df['sentiment_score'].values,
            'confidence': [random.uniform(0.6, 0.95) for _ in range(len(df))]
        }, index=df['timestamp'])
        sentiment_df.index = pd.to_datetime(sentiment_df.index)
        sentiment_data_dict[symbol] = sentiment_df
    
    # Initialize the sentiment provider with predefined data
    mock_sentiment_provider = MockSentimentProvider(predefined_data=sentiment_data_dict)
    
    # Create mock LLM oversight client
    llm_client = MockLLMOversightClient()
    
    # Configure market regime detection
    regime_config = MarketRegimeConfig(
        lookback_period=60,
        short_lookback=20,
        medium_lookback=50,
        long_lookback=120,
        volatility_window=20,
        regime_change_threshold=0.15,
        correlation_threshold=0.6
    )
    
    # Create the adaptive health orchestrator with mock oversight
    orchestrator = AdaptiveHealthOrchestrator(
        health_monitor=None,  # Will be created automatically
        log_dir="./logs",
        heartbeat_interval=5.0,
        monitor_components=True,
        regime_config=regime_config,
        temporal_pattern_enabled=True,
        adaptation_interval_minutes=10,  # Faster adaptation for testing
        base_portfolio_risk=0.02,
        max_position_size=0.20,
        enable_advanced_risk_management=True,
        enable_volatility_clustering=True,
        enable_correlation_optimization=True,
        enable_risk_parity=True,
        enable_llm_oversight=True,
        llm_oversight_level="advise"
    )
    
    # Replace the real oversight client with our mock
    orchestrator.oversight_client = llm_client
    
    # Register data sources
    logger.info("Registering mock data sources with orchestrator")
    orchestrator.register_market_data_source("price_data", mock_data_provider)
    orchestrator.register_market_data_source("sentiment_data", mock_sentiment_provider)
    
    # Create and register specialized agents
    logger.info("Creating and registering specialized agents")
    
    # Technical analysis agent
    tech_agent = BaseAgent(
        agent_id="technical_analysis",
        name="Technical Analysis Agent",
        agent_role=AgentRole.SPECIALIZED_TECHNICAL,
        agent_type="technical_analysis",
        symbols=symbols
    )
    
    # Sentiment analysis agent
    sentiment_agent = BaseAgent(
        agent_id="sentiment_analysis",
        name="Sentiment Analysis Agent",
        agent_role=AgentRole.SPECIALIZED_SENTIMENT,
        agent_type="sentiment_analysis",
        symbols=symbols
    )
    
    # Register agents with orchestrator
    orchestrator.register_agent(tech_agent)
    orchestrator.register_agent(sentiment_agent)
    
    # Create strategy instances
    tech_strategy = MACrossoverStrategy(
        symbols=symbols,
        fast_period=10,
        slow_period=30,
        risk_pct=0.02,
        max_position_pct=0.2
    )
    sentiment_strategy = SentimentStrategy(
        name="News Sentiment Strategy",
        config={
            "symbols": symbols,
            "sentiment_threshold": 0.2,
            "confidence_threshold": 0.6,
            "window_size": 14,
            "signal_processing": {
                "sentiment_filter": "ema",
                "sentiment_filter_window": 5
            }
        }
    )
    
    # Register strategies
    orchestrator.register_strategy("technical", tech_strategy)
    orchestrator.register_strategy("sentiment", sentiment_strategy)
    
    # Create and register meta strategy for signal aggregation
    meta_strategy = DynamicAggregationMetaStrategy(
        strategies={
            "technical": 0.6,
            "sentiment": 0.4
        },
        adaptation_weight=0.2  # How much to adapt weights based on performance
    )
    orchestrator.register_meta_strategy(meta_strategy)
    
    # Set initial portfolio state for better testing
    orchestrator.update_portfolio_state(
        portfolio_value=100000.0,
        positions={symbol: {"size": 0, "entry_price": 0} for symbol in symbols},
        drawdown=0.0
    )
    
    # Run the orchestrator in test mode
    logger.info(f"Starting autonomous trading test for {test_duration_seconds} seconds")
    
    start_time = time.time()
    cycle_count = 0
    
    try:
        while time.time() - start_time < test_duration_seconds:
            cycle_count += 1
            logger.info(f"\n--- Autonomous Trading Cycle {cycle_count} ---")
            
            # Run a single orchestration cycle
            results = orchestrator.run_cycle()
            
            # Log the results
            if results:
                logger.info(f"Detected Market Regime: {results.get('market_regime', 'Unknown')}")
                logger.info(f"Volatility Regime: {results.get('volatility_regime', 'Unknown')}")
                
                if 'signals' in results:
                    logger.info(f"Trading Signals Generated: {results['signals']}")
                
                if 'adaptations' in results:
                    logger.info("Adaptations Applied:")
                    for adaptation in results.get('adaptations', []):
                        logger.info(f"  - {adaptation}")
                
                if 'llm_oversight' in results:
                    oversight = results['llm_oversight']
                    logger.info(f"LLM Oversight: {oversight.get('action', 'Unknown')} - {oversight.get('reason', 'No reason')}")
                
                if 'health_status' in results:
                    health = results['health_status'] 
                    logger.info(f"System Health: {health}")
                    
                    # Check for any health issues
                    if 'alerts' in health and health['alerts']:
                        logger.warning(f"Health Alerts: {len(health['alerts'])} issues detected")
                        for alert in health['alerts']:
                            logger.warning(f"  - {alert}")
                    
                    # Log recovery actions if any
                    if 'recovery_actions' in health and health['recovery_actions']:
                        logger.info("Autonomous Recovery Actions:")
                        for action in health['recovery_actions']:
                            logger.info(f"  - {action}")
            
            # Pause between cycles to simulate real-time trading
            time.sleep(2)
    
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
    except Exception as e:
        logger.error(f"Error during autonomous trading test: {e}", exc_info=True)
    finally:
        # Report final status
        logger.info("\n=== Autonomous Trading Test Summary ===")
        logger.info(f"Completed {cycle_count} autonomous trading cycles")
        
        # Get regime history
        regimes = orchestrator.get_regime_history(days=60)
        logger.info(f"Market Regime History: {regimes}")
        
        # Get current regime
        current_regime = orchestrator.get_current_regime()
        logger.info(f"Final Market Regime: {current_regime}")
        
        # Get adaptation history
        logger.info("Strategy Adaptations Summary:")
        adaptation_stats = orchestrator.get_adaptation_statistics() if hasattr(orchestrator, 'get_adaptation_statistics') else {}
        for key, value in adaptation_stats.items():
            logger.info(f"  {key}: {value}")
        
        logger.info("Test completed successfully")

if __name__ == "__main__":
    try:
        run_autonomous_system_test(
            symbols=['BTC/USD', 'ETH/USD', 'AAPL', 'MSFT'], 
            test_duration_seconds=30
        )
    except Exception as e:
        logger.error(f"Autonomous test failed: {e}", exc_info=True)
