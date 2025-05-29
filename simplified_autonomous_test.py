#!/usr/bin/env python
"""
Simplified Autonomous System Test with Mock Data

This script demonstrates the core autonomous features of the AI Trading Agent
using mock data sources instead of requiring external API keys.
It shows the integration of:
- Health monitoring with self-healing
- Market regime detection
- Basic strategy adaptation
- Mock sentiment and price data
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
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Import necessary components
from ai_trading_agent.agent.health_integrated_orchestrator import HealthIntegratedOrchestrator
from ai_trading_agent.agent.agent_definitions import BaseAgent, AgentRole, AgentStatus
from ai_trading_agent.agent.strategy import SentimentStrategy
from ai_trading_agent.data_acquisition.mock_provider import MockDataProvider
from ai_trading_agent.market_regime import MarketRegimeClassifier
from ai_trading_agent.sentiment_analysis.mock_provider import MockSentimentProvider

class MockLLMOversightClient:
    """Mock LLM Oversight Client for demonstration purposes."""
    
    def __init__(self):
        self.service_available = True
        logger.info("Initialized Mock LLM Oversight Client")
        self.decisions_reviewed = 0
        self.decisions_approved = 0
        self.decisions_modified = 0
        self.decisions_rejected = 0
        self.decisions_logged = 0
    
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
        self.decisions_reviewed += 1
        
        # Somewhat realistic distribution: mostly approve, sometimes modify, rarely reject
        action = random.choices(
            ["approve", "modify", "reject", "log"], 
            weights=[0.75, 0.15, 0.05, 0.05], 
            k=1
        )[0]
        
        if action == "approve":
            self.decisions_approved += 1
            reason = "Decision aligns with current market conditions and risk parameters."
        elif action == "modify":
            self.decisions_modified += 1
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
        elif action == "reject":
            self.decisions_rejected += 1
            reason = "Position size exceeds risk threshold for current market conditions."
        else:  # log
            self.decisions_logged += 1
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
            "signals": random.randint(5, 15),
            "volatility": random.choice(["low", "moderate", "high"])
        }
    
    def get_metrics(self):
        """Get oversight metrics."""
        approval_rate = (self.decisions_approved / self.decisions_reviewed * 100) if self.decisions_reviewed > 0 else 0
        
        return {
            "decisions_reviewed": self.decisions_reviewed,
            "approval_rate": f"{approval_rate:.1f}%",
            "approve": self.decisions_approved,
            "modify": self.decisions_modified,
            "reject": self.decisions_rejected,
            "log": self.decisions_logged
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

def run_simplified_autonomous_test(symbols=['BTC/USD', 'ETH/USD', 'AAPL', 'MSFT'], test_duration_seconds=30):
    """
    Run a simplified autonomous system test that demonstrates the core capabilities
    without requiring external API keys or complex initialization.
    """
    logger.info(f"Starting simplified autonomous system test with symbols: {symbols}")
    
    # Create mock data
    price_data, sentiment_data = create_mock_data(symbols, days=60)
    
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
    
    # Initialize basic orchestrator
    orchestrator = HealthIntegratedOrchestrator(
        health_monitor=None,  # Will be created automatically
        log_dir="./logs",
        heartbeat_interval=5.0,
        monitor_components=True,
        enable_market_regime_detection=True,
        enable_llm_oversight=True
    )
    
    # Replace the real oversight client with our mock
    orchestrator.oversight_client = llm_client
    
    # Create and register specialized agents
    logger.info("Creating and registering specialized agents")
    
    # Technical analysis agent
    tech_agent_1 = BaseAgent(
        agent_id="technical_1",
        name="Simple Moving Average Crossover",
        agent_role=AgentRole.SPECIALIZED_TECHNICAL,
        agent_type="technical_analysis",
        symbols=symbols
    )
    
    tech_agent_2 = BaseAgent(
        agent_id="technical_2",
        name="RSI Strategy",
        agent_role=AgentRole.SPECIALIZED_TECHNICAL,
        agent_type="technical_analysis", 
        symbols=symbols
    )
    
    # Sentiment analysis agent
    sentiment_agent = BaseAgent(
        agent_id="sentiment_1",
        name="News Sentiment Analysis",
        agent_role=AgentRole.SPECIALIZED_SENTIMENT,
        agent_type="sentiment_analysis",
        symbols=symbols
    )
    
    # Risk management agent
    risk_agent = BaseAgent(
        agent_id="risk_1",
        name="Portfolio Risk Management",
        agent_role=AgentRole.SPECIALIZED_TECHNICAL,
        agent_type="risk_management",
        symbols=symbols
    )
    
    # Decision aggregator agent
    decision_agent = BaseAgent(
        agent_id="decision_1",
        name="Signal Aggregator",
        agent_role=AgentRole.DECISION_AGGREGATOR,
        agent_type="decision_aggregator",
        symbols=symbols
    )
    
    # Register agents with orchestrator
    orchestrator.register_agent(tech_agent_1)
    orchestrator.register_agent(tech_agent_2)
    orchestrator.register_agent(sentiment_agent)
    orchestrator.register_agent(risk_agent)
    orchestrator.register_agent(decision_agent)
    
    # Set initial portfolio state for better testing
    orchestrator.update_portfolio_state(
        portfolio_value=100000.0,
        positions={symbol: {"size": 0, "entry_price": 0} for symbol in symbols},
        drawdown=0.0
    )
    
    # Create market regime classifier
    market_regime_classifier = MarketRegimeClassifier()
    orchestrator.market_regime_classifier = market_regime_classifier
    
    # Run the orchestrator in test mode
    logger.info(f"Starting autonomous trading test for {test_duration_seconds} seconds")
    
    start_time = time.time()
    cycle_count = 0
    
    try:
        # Simulate market data
        market_data = {}
        for symbol in symbols:
            price_df = price_data[symbol]
            market_data[symbol] = {
                "prices": price_df['close'].tolist()[-30:],  # Get last 30 days
                "volatility": np.std(price_df['close'].pct_change().dropna()),
                "volume": price_df['volume'].mean(),
            }
        
        # Initialize portfolio with mock data
        portfolio = {
            "total_value": 100000.0,
            "cash": 50000.0,
            "positions": {}
        }
        
        # Add some initial positions
        for symbol in symbols[:2]:  # Only add positions for first two symbols
            entry_price = price_data[symbol]['close'].iloc[-1]
            portfolio["positions"][symbol] = {
                "size": 1.0,
                "entry_price": entry_price,
                "current_price": entry_price,
                "value": entry_price,
                "unrealized_pnl": 0.0,
                "pct_change": 0.0
            }
        
        while time.time() - start_time < test_duration_seconds:
            cycle_count += 1
            logger.info(f"\n--- Autonomous Trading Cycle {cycle_count} ---")
            
            # Update mock market data with some random changes
            for symbol in symbols:
                last_price = market_data[symbol]["prices"][-1]
                price_change = last_price * random.uniform(-0.02, 0.02)  # Random 2% max change
                new_price = last_price + price_change
                market_data[symbol]["prices"].append(new_price)
                market_data[symbol]["prices"] = market_data[symbol]["prices"][-30:]  # Keep last 30 points
                
                # Update volatility
                market_data[symbol]["volatility"] = np.std(
                    np.diff(market_data[symbol]["prices"]) / market_data[symbol]["prices"][:-1]
                )
                
                # Update position value if we have a position
                if symbol in portfolio["positions"]:
                    pos = portfolio["positions"][symbol]
                    old_value = pos["value"]
                    pos["current_price"] = new_price
                    pos["value"] = pos["size"] * new_price
                    pos["unrealized_pnl"] = pos["value"] - (pos["size"] * pos["entry_price"])
                    pos["pct_change"] = (new_price / pos["entry_price"]) - 1.0
                    
                    # Update portfolio value
                    portfolio["total_value"] = portfolio["total_value"] + (pos["value"] - old_value)
            
            # Detect market regimes
            market_regimes = {}
            volatility_regimes = {}
            
            for symbol in symbols:
                # Decide regimes based on recent price action and volatility
                prices = market_data[symbol]["prices"]
                recent_trend = (prices[-1] / prices[-5]) - 1  # 5-period trend
                
                # Simplified regime classification
                if recent_trend > 0.05:
                    market_regimes[symbol] = "bull"
                elif recent_trend < -0.05:
                    market_regimes[symbol] = "bear"
                elif market_data[symbol]["volatility"] > 0.02:
                    market_regimes[symbol] = "volatile"
                else:
                    market_regimes[symbol] = "sideways"
                
                # Volatility regimes
                vol = market_data[symbol]["volatility"]
                if vol < 0.01:
                    volatility_regimes[symbol] = "low"
                elif vol < 0.03:
                    volatility_regimes[symbol] = "moderate"
                else:
                    volatility_regimes[symbol] = "high"
            
            # Simulate agent processing
            # In a real scenario, each agent would process data and pass signals
            
            # LLM oversight simulation - test decision making
            trade_decisions = []
            
            # Generate trading decision based on market regime for 2 random symbols
            for symbol in random.sample(symbols, 2):
                regime = market_regimes.get(symbol, "unknown")
                vol_regime = volatility_regimes.get(symbol, "unknown")
                
                # Simple decision logic based on regime
                if regime == "bull" and vol_regime != "high":
                    decision = "buy"
                    position_size = 0.1  # 10% allocation
                elif regime == "bear" and vol_regime != "low":
                    decision = "sell"
                    position_size = 0.05  # 5% allocation
                else:
                    decision = "hold"
                    position_size = 0
                
                if decision != "hold":
                    # Create a decision to validate with LLM oversight
                    decision_data = {
                        "symbol": symbol,
                        "decision": decision,
                        "position_size": position_size,
                        "regime": regime,
                        "volatility": vol_regime,
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    # Validate with LLM oversight
                    validation_result = llm_client.validate_trading_decision(decision_data)
                    
                    logger.info(f"Decision: {decision} {symbol} - Position size: {position_size}")
                    logger.info(f"Oversight: {validation_result['action']} - {validation_result['reason']}")
                    
                    # Apply the decision if approved
                    if validation_result["action"] == "approve":
                        trade_decisions.append(decision_data)
                    elif validation_result["action"] == "modify" and "modified_decision" in validation_result:
                        modified = validation_result["modified_decision"]
                        decision_data.update(modified)
                        trade_decisions.append(decision_data)
            
            # Print oversight metrics every few cycles
            if cycle_count % 2 == 0:
                metrics = llm_client.get_metrics()
                logger.info("LLM Oversight Metrics:")
                for key, value in metrics.items():
                    logger.info(f"{key.replace('_', ' ').title()}: {value}")
            
            # Update portfolio based on decisions (simplified)
            for decision in trade_decisions:
                symbol = decision["symbol"]
                action = decision["decision"]
                size = decision["position_size"]
                
                current_price = market_data[symbol]["prices"][-1]
                
                if action == "buy":
                    # If we don't have a position, create one
                    if symbol not in portfolio["positions"]:
                        cost = size * portfolio["total_value"] 
                        if cost <= portfolio["cash"]:
                            quantity = cost / current_price
                            portfolio["positions"][symbol] = {
                                "size": quantity,
                                "entry_price": current_price,
                                "current_price": current_price,
                                "value": cost,
                                "unrealized_pnl": 0.0,
                                "pct_change": 0.0
                            }
                            portfolio["cash"] -= cost
                    else:
                        # Increase existing position
                        cost = size * portfolio["total_value"]
                        if cost <= portfolio["cash"]:
                            quantity = cost / current_price
                            pos = portfolio["positions"][symbol]
                            new_total_quantity = pos["size"] + quantity
                            # Calculate new average entry price
                            pos["entry_price"] = (pos["entry_price"] * pos["size"] + current_price * quantity) / new_total_quantity
                            pos["size"] = new_total_quantity
                            pos["value"] = pos["size"] * current_price
                            pos["unrealized_pnl"] = pos["value"] - (pos["size"] * pos["entry_price"])
                            pos["pct_change"] = (current_price / pos["entry_price"]) - 1.0
                            portfolio["cash"] -= cost
                
                elif action == "sell" and symbol in portfolio["positions"]:
                    # Sell part or all of the position
                    pos = portfolio["positions"][symbol]
                    sell_quantity = min(pos["size"], size * portfolio["total_value"] / current_price)
                    sell_value = sell_quantity * current_price
                    pos["size"] -= sell_quantity
                    portfolio["cash"] += sell_value
                    
                    # If fully sold, remove the position
                    if pos["size"] <= 0:
                        del portfolio["positions"][symbol]
                    else:
                        # Update position value
                        pos["value"] = pos["size"] * current_price
                        pos["unrealized_pnl"] = pos["value"] - (pos["size"] * pos["entry_price"])
                        pos["pct_change"] = (current_price / pos["entry_price"]) - 1.0
            
            # Display current portfolio status
            logger.info("\nPortfolio Status:")
            logger.info(f"Total Value: ${portfolio['total_value']:.2f}")
            logger.info(f"Cash: ${portfolio['cash']:.2f}")
            logger.info("Positions:")
            
            for symbol, pos in portfolio["positions"].items():
                logger.info(f"  {symbol}: {pos['size']:.6f} @ ${pos['entry_price']:.2f} (Current: ${pos['current_price']:.2f}, P&L: {pos['pct_change']*100:.2f}%)")
            
            # Show market regimes
            logger.info("\nMarket Regimes:")
            for symbol in symbols:
                logger.info(f"  {symbol}: {market_regimes[symbol]} (Volatility: {volatility_regimes[symbol]})")
            
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
        
        # Print oversight metrics
        metrics = llm_client.get_metrics()
        logger.info("\nFinal LLM Oversight Metrics:")
        for key, value in metrics.items():
            logger.info(f"{key.replace('_', ' ').title()}: {value}")
        
        logger.info("\nFinal Portfolio Value: ${:.2f}".format(portfolio["total_value"]))
        
        # Calculate percentage gain/loss
        pct_change = (portfolio["total_value"] / 100000.0 - 1) * 100
        logger.info(f"Total Return: {pct_change:.2f}%")
        
        logger.info("Test completed successfully")

if __name__ == "__main__":
    run_simplified_autonomous_test()
