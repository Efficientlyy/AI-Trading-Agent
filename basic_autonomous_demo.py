#!/usr/bin/env python
"""
Basic Autonomous Demo with Mock Data

This script provides a basic demonstration of the AI Trading Agent using mock data,
focusing on the core autonomous capabilities without requiring external API keys or
complex initialization of all components.
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

class MockHealthMonitor:
    """A simple mock health monitoring system."""
    
    def __init__(self):
        logger.info("Health monitoring system initialized")
        self.component_health = {}
        self.alerts = []
    
    def register_component(self, component_id, description=None):
        """Register a component to be monitored."""
        self.component_health[component_id] = {
            "status": "HEALTHY",
            "last_heartbeat": datetime.now(),
            "alerts": []
        }
        logger.info(f"Registered component for health monitoring: {component_id}")
    
    def record_heartbeat(self, component_id):
        """Record a heartbeat for a component."""
        if component_id in self.component_health:
            self.component_health[component_id]["last_heartbeat"] = datetime.now()
    
    def add_alert(self, component_id, message, severity="WARNING"):
        """Add an alert for a component."""
        self.alerts.append({
            "component_id": component_id,
            "message": message,
            "severity": severity,
            "timestamp": datetime.now()
        })
        
        if severity == "ERROR":
            self.component_health[component_id]["status"] = "UNHEALTHY"
        elif severity == "WARNING" and self.component_health[component_id]["status"] == "HEALTHY":
            self.component_health[component_id]["status"] = "DEGRADED"
    
    def get_health_status(self):
        """Get the overall health status."""
        statuses = [component["status"] for component in self.component_health.values()]
        if "UNHEALTHY" in statuses:
            return "UNHEALTHY"
        elif "DEGRADED" in statuses:
            return "DEGRADED"
        else:
            return "HEALTHY"

class MockMarketRegimeClassifier:
    """A simple mock market regime classifier."""
    
    def __init__(self):
        logger.info("Market Regime Classifier initialized")
        self.regimes = {
            "bull": {"volatility": "low", "momentum": "positive", "correlation": "high"},
            "bear": {"volatility": "high", "momentum": "negative", "correlation": "high"},
            "sideways": {"volatility": "low", "momentum": "neutral", "correlation": "low"},
            "volatile": {"volatility": "high", "momentum": "mixed", "correlation": "mixed"}
        }
    
    def classify_regime(self, prices, timeframe="1d"):
        """Classify the market regime based on price history."""
        if not prices or len(prices) < 10:
            return {
                "regime": "unknown",
                "confidence": 0.0,
                "details": {}
            }
        
        # Calculate some basic metrics
        returns = [prices[i]/prices[i-1] - 1 for i in range(1, len(prices))]
        volatility = np.std(returns) * np.sqrt(252)  # Annualized
        trend = np.mean(returns) * 252  # Annualized
        
        # Determine regime based on simple thresholds
        if volatility > 0.20:  # High volatility
            if trend > 0.05:
                regime = "volatile_bull"
            elif trend < -0.05:
                regime = "volatile_bear"
            else:
                regime = "volatile"
        else:  # Lower volatility
            if trend > 0.10:
                regime = "bull"
            elif trend < -0.10:
                regime = "bear"
            else:
                regime = "sideways"
        
        return {
            "regime": regime,
            "confidence": 0.7 + random.random() * 0.3,  # Random confidence 0.7-1.0
            "details": {
                "volatility": volatility,
                "trend": trend
            }
        }

class MockLLMOversight:
    """A simple mock LLM oversight system."""
    
    def __init__(self):
        logger.info("LLM Oversight System initialized")
        self.decisions_reviewed = 0
        self.decisions_approved = 0
        self.decisions_modified = 0
        self.decisions_rejected = 0
    
    def review_decision(self, decision_data):
        """Review a trading decision."""
        self.decisions_reviewed += 1
        
        # Implement a simple decision review process that's mostly approving
        random_value = random.random()
        
        if random_value < 0.75:  # 75% approval rate
            self.decisions_approved += 1
            return {"action": "approve", "reason": "Decision aligns with market conditions"}
        elif random_value < 0.90:  # 15% modification rate
            self.decisions_modified += 1
            # Modify position size
            modified_size = decision_data.get("position_size", 0) * 0.8  # Reduce by 20%
            return {
                "action": "modify", 
                "reason": "Reducing position size due to current volatility",
                "modifications": {"position_size": modified_size}
            }
        else:  # 10% rejection rate
            self.decisions_rejected += 1
            return {"action": "reject", "reason": "Risk parameters exceeded"}
    
    def get_metrics(self):
        """Get oversight metrics."""
        return {
            "decisions_reviewed": self.decisions_reviewed,
            "approved": self.decisions_approved,
            "modified": self.decisions_modified,
            "rejected": self.decisions_rejected,
            "approval_rate": f"{(self.decisions_approved / max(1, self.decisions_reviewed)) * 100:.1f}%"
        }

class AdaptiveTradingSystem:
    """A simplified adaptive trading system demonstration."""
    
    def __init__(self):
        logger.info("Adaptive Trading System initialized")
        self.health_monitor = MockHealthMonitor()
        self.market_classifier = MockMarketRegimeClassifier()
        self.llm_oversight = MockLLMOversight()
        
        self.agents = {}
        self.portfolio = {
            "cash": 100000.0,
            "total_value": 100000.0,
            "positions": {},
            "history": []
        }
        
        # Market data storage
        self.market_data = {}
        
        # Agent information
        self.agent_types = {
            "technical_analysis": {
                "description": "Analyzes price patterns and indicators",
                "default_weight": 0.6
            },
            "sentiment_analysis": {
                "description": "Analyzes news sentiment",
                "default_weight": 0.4
            },
            "risk_management": {
                "description": "Manages portfolio risk",
                "default_weight": 1.0  # Can override other signals
            },
            "decision": {
                "description": "Aggregates signals and makes trading decisions",
                "default_weight": 1.0
            }
        }
    
    def initialize_agent(self, agent_id, agent_type, name=None):
        """Initialize a trading agent."""
        if agent_type not in self.agent_types:
            raise ValueError(f"Unknown agent type: {agent_type}")
        
        if not name:
            name = f"{agent_type.replace('_', ' ').title()} Agent {agent_id.split('_')[-1]}"
        
        agent = {
            "id": agent_id,
            "type": agent_type,
            "name": name,
            "status": "INITIALIZED",
            "last_update": datetime.now(),
            "signals": [],
            "metrics": {
                "signal_count": 0,
                "accuracy": 0.5,  # Initial accuracy assumption
                "latency": 0.0
            }
        }
        
        self.agents[agent_id] = agent
        
        # Register with health monitor
        self.health_monitor.register_component(agent_id)
        
        logger.info(f"Initialized {agent_type} agent: {agent_id}")
        return agent
    
    def update_market_data(self, symbol, price, volume=None, timestamp=None):
        """Update market data for a symbol."""
        if symbol not in self.market_data:
            self.market_data[symbol] = {
                "prices": [],
                "volumes": [],
                "timestamps": [],
                "regimes": {}
            }
        
        curr_time = timestamp or datetime.now()
        
        self.market_data[symbol]["prices"].append(price)
        self.market_data[symbol]["volumes"].append(volume or random.randint(1000, 10000))
        self.market_data[symbol]["timestamps"].append(curr_time)
        
        # Keep only the last 100 data points
        if len(self.market_data[symbol]["prices"]) > 100:
            self.market_data[symbol]["prices"] = self.market_data[symbol]["prices"][-100:]
            self.market_data[symbol]["volumes"] = self.market_data[symbol]["volumes"][-100:]
            self.market_data[symbol]["timestamps"] = self.market_data[symbol]["timestamps"][-100:]
        
        # Update market regime if we have enough data
        if len(self.market_data[symbol]["prices"]) >= 10:
            regime_info = self.market_classifier.classify_regime(self.market_data[symbol]["prices"])
            self.market_data[symbol]["regimes"] = regime_info
    
    def generate_technical_signal(self, agent_id, symbol):
        """Generate a technical analysis signal."""
        if symbol not in self.market_data or len(self.market_data[symbol]["prices"]) < 10:
            return None
        
        agent = self.agents.get(agent_id)
        if not agent or agent["type"] != "technical_analysis":
            return None
        
        # Simple moving average crossover
        prices = self.market_data[symbol]["prices"]
        short_ma = sum(prices[-5:]) / 5
        long_ma = sum(prices[-20:]) / 20
        
        if short_ma > long_ma:
            signal = {"direction": "buy", "strength": 0.6 + random.random() * 0.4}
        elif short_ma < long_ma:
            signal = {"direction": "sell", "strength": 0.6 + random.random() * 0.4}
        else:
            signal = {"direction": "hold", "strength": 0.5}
        
        signal.update({
            "symbol": symbol,
            "agent_id": agent_id,
            "timestamp": datetime.now(),
            "type": "technical",
            "indicators": {
                "short_ma": short_ma,
                "long_ma": long_ma
            }
        })
        
        # Record the signal
        agent["signals"].append(signal)
        agent["metrics"]["signal_count"] += 1
        
        # Update heartbeat
        self.health_monitor.record_heartbeat(agent_id)
        
        return signal
    
    def generate_sentiment_signal(self, agent_id, symbol):
        """Generate a sentiment analysis signal."""
        agent = self.agents.get(agent_id)
        if not agent or agent["type"] != "sentiment_analysis":
            return None
        
        # Generate a random sentiment score with some bias based on recent price action
        if symbol in self.market_data and len(self.market_data[symbol]["prices"]) >= 5:
            prices = self.market_data[symbol]["prices"]
            recent_change = prices[-1] / prices[-5] - 1
            
            # Add a slight bias based on recent price movement
            bias = 0.3 * np.sign(recent_change)
            
            # Add randomness (-0.7 to +0.7) to the sentiment
            randomness = (random.random() - 0.5) * 1.4
            
            sentiment_score = bias + randomness
            # Clamp to [-1, 1]
            sentiment_score = max(min(sentiment_score, 1.0), -1.0)
            
            if sentiment_score > 0.3:
                direction = "buy"
                strength = 0.5 + sentiment_score / 2
            elif sentiment_score < -0.3:
                direction = "sell"
                strength = 0.5 + abs(sentiment_score) / 2
            else:
                direction = "hold"
                strength = 0.5
            
            signal = {
                "direction": direction,
                "strength": strength,
                "symbol": symbol,
                "agent_id": agent_id,
                "timestamp": datetime.now(),
                "type": "sentiment",
                "sentiment_score": sentiment_score
            }
            
            # Record the signal
            agent["signals"].append(signal)
            agent["metrics"]["signal_count"] += 1
            
            # Update heartbeat
            self.health_monitor.record_heartbeat(agent_id)
            
            return signal
        
        return None
    
    def generate_risk_signal(self, agent_id, symbol):
        """Generate a risk management signal."""
        agent = self.agents.get(agent_id)
        if not agent or agent["type"] != "risk_management":
            return None
        
        # Simplified risk check
        position = self.portfolio["positions"].get(symbol, {})
        position_value = position.get("value", 0)
        
        # Check if position exceeds 20% of portfolio
        if position_value > 0 and position_value / self.portfolio["total_value"] > 0.2:
            signal = {
                "direction": "reduce",
                "strength": 0.8,
                "symbol": symbol,
                "agent_id": agent_id,
                "timestamp": datetime.now(),
                "type": "risk",
                "reason": "Position exceeds 20% of portfolio"
            }
        else:
            signal = None
        
        if signal:
            agent["signals"].append(signal)
            agent["metrics"]["signal_count"] += 1
        
        # Update heartbeat
        self.health_monitor.record_heartbeat(agent_id)
        
        return signal
    
    def aggregate_signals(self, agent_id, symbol):
        """Aggregate signals from various agents to make a decision."""
        agent = self.agents.get(agent_id)
        if not agent or agent["type"] != "decision":
            return None
        
        # Collect all recent signals for this symbol
        all_signals = []
        for a_id, a_info in self.agents.items():
            if a_id == agent_id:
                continue  # Skip the decision agent itself
            
            # Get recent signals (last 5)
            signals = [s for s in a_info["signals"] if s["symbol"] == symbol][-5:]
            all_signals.extend(signals)
        
        if not all_signals:
            return None
        
        # Check for risk signals first (they have priority)
        risk_signals = [s for s in all_signals if s["type"] == "risk"]
        if risk_signals:
            decision = risk_signals[0]  # Use the most recent risk signal
            decision["source"] = "risk_override"
            return decision
        
        # Weighted aggregation of technical and sentiment signals
        tech_signals = [s for s in all_signals if s["type"] == "technical"]
        sent_signals = [s for s in all_signals if s["type"] == "sentiment"]
        
        # Default weights
        tech_weight = 0.6
        sent_weight = 0.4
        
        # Adjust weights based on market regime if available
        if symbol in self.market_data and "regimes" in self.market_data[symbol]:
            regime = self.market_data[symbol]["regimes"].get("regime", "unknown")
            if regime == "volatile" or regime == "volatile_bear":
                # In volatile markets, rely more on technical signals
                tech_weight = 0.8
                sent_weight = 0.2
            elif regime == "bull":
                # In bull markets, sentiment is more important
                tech_weight = 0.4
                sent_weight = 0.6
        
        # Calculate weighted signal
        buy_score = 0
        sell_score = 0
        signal_count = 0
        
        for signal in tech_signals:
            if signal["direction"] == "buy":
                buy_score += signal["strength"] * tech_weight
                signal_count += 1
            elif signal["direction"] == "sell":
                sell_score += signal["strength"] * tech_weight
                signal_count += 1
        
        for signal in sent_signals:
            if signal["direction"] == "buy":
                buy_score += signal["strength"] * sent_weight
                signal_count += 1
            elif signal["direction"] == "sell":
                sell_score += signal["strength"] * sent_weight
                signal_count += 1
        
        if signal_count == 0:
            return None
        
        # Normalize scores
        buy_score /= signal_count
        sell_score /= signal_count
        
        # Make decision
        if buy_score > 0.6 and buy_score > sell_score:
            direction = "buy"
            strength = buy_score
        elif sell_score > 0.6 and sell_score > buy_score:
            direction = "sell"
            strength = sell_score
        else:
            direction = "hold"
            strength = 0.5
        
        decision = {
            "direction": direction,
            "strength": strength,
            "symbol": symbol,
            "agent_id": agent_id,
            "timestamp": datetime.now(),
            "type": "decision",
            "buy_score": buy_score,
            "sell_score": sell_score,
            "signal_count": signal_count,
            "source": "weighted_consensus"
        }
        
        # Record the decision
        agent["signals"].append(decision)
        agent["metrics"]["signal_count"] += 1
        
        # Update heartbeat
        self.health_monitor.record_heartbeat(agent_id)
        
        return decision
    
    def execute_decision(self, decision):
        """Execute a trading decision."""
        if not decision or decision["direction"] == "hold":
            return False
        
        symbol = decision["symbol"]
        direction = decision["direction"]
        
        # Calculate position size based on signal strength
        position_size = 0.1 * decision["strength"]  # 10% of portfolio max
        
        if direction == "buy":
            # Check if we have this symbol already
            if symbol in self.portfolio["positions"]:
                # Add to existing position
                position = self.portfolio["positions"][symbol]
                current_price = self.market_data[symbol]["prices"][-1]
                
                # Calculate how many to buy
                value_to_add = self.portfolio["total_value"] * position_size
                additional_units = value_to_add / current_price
                
                if value_to_add <= self.portfolio["cash"]:
                    # Update position
                    position["units"] += additional_units
                    position["value"] += value_to_add
                    position["average_price"] = (position["value"] / position["units"])
                    
                    # Update portfolio
                    self.portfolio["cash"] -= value_to_add
                    logger.info(f"Added to position: {symbol} - {additional_units:.4f} units at ${current_price:.2f}")
                    return True
            else:
                # New position
                current_price = self.market_data[symbol]["prices"][-1]
                value_to_add = self.portfolio["total_value"] * position_size
                units = value_to_add / current_price
                
                if value_to_add <= self.portfolio["cash"]:
                    # Create new position
                    self.portfolio["positions"][symbol] = {
                        "units": units,
                        "average_price": current_price,
                        "value": value_to_add,
                        "entry_time": datetime.now()
                    }
                    
                    # Update portfolio
                    self.portfolio["cash"] -= value_to_add
                    logger.info(f"New position: {symbol} - {units:.4f} units at ${current_price:.2f}")
                    return True
        
        elif direction == "sell" or direction == "reduce":
            # Check if we have this symbol
            if symbol in self.portfolio["positions"]:
                position = self.portfolio["positions"][symbol]
                current_price = self.market_data[symbol]["prices"][-1]
                
                # Calculate how many to sell
                if direction == "sell":
                    units_to_sell = position["units"]  # Sell all
                else:  # reduce
                    units_to_sell = position["units"] * position_size  # Sell a portion
                
                value_to_liquidate = units_to_sell * current_price
                
                # Update position
                position["units"] -= units_to_sell
                position["value"] -= value_to_liquidate
                
                # Update portfolio
                self.portfolio["cash"] += value_to_liquidate
                
                # Remove position if fully sold
                if position["units"] <= 0 or direction == "sell":
                    del self.portfolio["positions"][symbol]
                    logger.info(f"Closed position: {symbol} - {units_to_sell:.4f} units at ${current_price:.2f}")
                else:
                    logger.info(f"Reduced position: {symbol} - {units_to_sell:.4f} units at ${current_price:.2f}")
                
                return True
        
        return False
    
    def update_portfolio_value(self):
        """Update the total portfolio value based on current prices."""
        position_value = 0
        
        for symbol, position in list(self.portfolio["positions"].items()):
            if symbol in self.market_data and self.market_data[symbol]["prices"]:
                current_price = self.market_data[symbol]["prices"][-1]
                current_value = position["units"] * current_price
                
                # Update position
                position["value"] = current_value
                position_value += current_value
            else:
                # If we don't have current price data, use the last known value
                position_value += position["value"]
        
        # Calculate total portfolio value
        self.portfolio["total_value"] = self.portfolio["cash"] + position_value
        
        # Add to history
        self.portfolio["history"].append({
            "timestamp": datetime.now(),
            "total_value": self.portfolio["total_value"],
            "cash": self.portfolio["cash"],
            "position_value": position_value
        })
    
    def run_cycle(self, symbols):
        """Run a single autonomous trading cycle."""
        # Update prices (in a real system, this would come from data feeds)
        for symbol in symbols:
            if symbol in self.market_data and self.market_data[symbol]["prices"]:
                last_price = self.market_data[symbol]["prices"][-1]
                # Generate a random price change (-2% to +2%)
                price_change = last_price * (1 + (random.random() - 0.5) * 0.04)
            else:
                # Initial price if we don't have data yet
                if "BTC" in symbol:
                    price_change = 40000 + random.random() * 2000
                elif "ETH" in symbol:
                    price_change = 2800 + random.random() * 200
                else:
                    price_change = 100 + random.random() * 20
            
            # Update market data
            self.update_market_data(symbol, price_change)
        
        # Generate signals from each technical agent
        for agent_id, agent in self.agents.items():
            if agent["type"] == "technical_analysis":
                for symbol in symbols:
                    self.generate_technical_signal(agent_id, symbol)
        
        # Generate signals from each sentiment agent
        for agent_id, agent in self.agents.items():
            if agent["type"] == "sentiment_analysis":
                for symbol in symbols:
                    self.generate_sentiment_signal(agent_id, symbol)
        
        # Generate risk signals
        for agent_id, agent in self.agents.items():
            if agent["type"] == "risk_management":
                for symbol in symbols:
                    self.generate_risk_signal(agent_id, symbol)
        
        # Generate decisions
        decisions = []
        for agent_id, agent in self.agents.items():
            if agent["type"] == "decision":
                for symbol in symbols:
                    decision = self.aggregate_signals(agent_id, symbol)
                    if decision and decision["direction"] != "hold":
                        # Review with LLM Oversight
                        review = self.llm_oversight.review_decision(decision)
                        
                        if review["action"] == "approve":
                            decisions.append(decision)
                        elif review["action"] == "modify" and "modifications" in review:
                            # Apply modifications
                            decision.update(review["modifications"])
                            decisions.append(decision)
        
        # Execute decisions
        for decision in decisions:
            self.execute_decision(decision)
        
        # Update portfolio value
        self.update_portfolio_value()
        
        # Return cycle results
        return {
            "decisions": decisions,
            "portfolio": {
                "total_value": self.portfolio["total_value"],
                "cash": self.portfolio["cash"],
                "positions": {k: v.copy() for k, v in self.portfolio["positions"].items()}
            },
            "market_regimes": {
                symbol: self.market_data[symbol].get("regimes", {}).get("regime", "unknown")
                for symbol in symbols
            },
            "health_status": self.health_monitor.get_health_status()
        }

def run_demo():
    """Run the automated trading demo."""
    # Set up the adaptive trading system
    trading_system = AdaptiveTradingSystem()
    
    # Define symbols to trade
    symbols = ["BTC/USD", "ETH/USD", "AAPL", "MSFT"]
    
    # Initialize agents
    trading_system.initialize_agent("technical_1", "technical_analysis")
    trading_system.initialize_agent("technical_2", "technical_analysis")
    trading_system.initialize_agent("sentiment_1", "sentiment_analysis")
    trading_system.initialize_agent("risk_1", "risk_management")
    trading_system.initialize_agent("decision_1", "decision")
    
    # Run trading cycles
    num_cycles = 10
    
    try:
        for cycle in range(1, num_cycles + 1):
            logger.info(f"\n--- Trading Cycle {cycle} ---")
            
            # Run a trading cycle
            results = trading_system.run_cycle(symbols)
            
            # Display results
            logger.info(f"Portfolio Value: ${results['portfolio']['total_value']:.2f}")
            logger.info(f"Cash: ${results['portfolio']['cash']:.2f}")
            
            if results['portfolio']['positions']:
                logger.info("Positions:")
                for symbol, pos in results['portfolio']['positions'].items():
                    logger.info(f"  {symbol}: {pos['units']:.6f} @ ${pos['average_price']:.2f} (Value: ${pos['value']:.2f})")
            
            logger.info("Market Regimes:")
            for symbol, regime in results['market_regimes'].items():
                logger.info(f"  {symbol}: {regime}")
            
            logger.info(f"System Health Status: {results['health_status']}")
            
            # Pause between cycles
            time.sleep(1)
        
        # Final summary
        logger.info("\n=== Trading Demo Complete ===")
        
        # Display LLM oversight metrics
        metrics = trading_system.llm_oversight.get_metrics()
        logger.info("LLM Oversight Metrics:")
        for key, value in metrics.items():
            logger.info(f"{key.replace('_', ' ').title()}: {value}")
        
        # Display final portfolio value
        initial_value = 100000.0
        final_value = trading_system.portfolio["total_value"]
        pct_change = (final_value / initial_value - 1) * 100
        
        logger.info(f"Initial Portfolio: ${initial_value:.2f}")
        logger.info(f"Final Portfolio: ${final_value:.2f}")
        logger.info(f"Total Return: {pct_change:.2f}%")
        
    except KeyboardInterrupt:
        logger.info("Demo interrupted by user")
    except Exception as e:
        logger.error(f"Error during demo: {e}", exc_info=True)

if __name__ == "__main__":
    run_demo()
