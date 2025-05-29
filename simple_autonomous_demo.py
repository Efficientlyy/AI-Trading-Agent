"""
Simple Autonomous AI Trading Agent Demo with Mock Data

This script demonstrates the core autonomous features of the AI Trading Agent
without requiring external API keys or complex dependencies.
"""

import logging
import random
import time
from datetime import datetime, timedelta
from enum import Enum

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# --------- Mock classes to simulate the AI Trading Agent system ---------

class MarketRegimeType(Enum):
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"
    TRENDING = "trending"
    
class VolatilityRegimeType(Enum):
    LOW = "low" 
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"
    
class AgentStatus(Enum):
    IDLE = "idle"
    RUNNING = "running"
    ERROR = "error"
    RECOVERING = "recovering"
    STANDBY = "standby"

class AgentType(Enum):
    TECHNICAL_ANALYSIS = "technical_analysis"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    RISK_MANAGEMENT = "risk_management"
    DECISION = "decision"
    EXECUTION = "execution"

class HealthStatus(Enum):
    HEALTHY = "healthy"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class OversightAction(Enum):
    APPROVE = "approve"
    MODIFY = "modify"
    REJECT = "reject"
    LOG = "log"

# --------- Core component classes ---------

class MockAgent:
    """Base agent implementation for the demo."""
    
    def __init__(self, agent_id, agent_type):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.status = AgentStatus.IDLE
        self.last_heartbeat = datetime.now()
        self.metrics = {}
        logger.info(f"Initialized {agent_type.value} agent: {agent_id}")
    
    def process(self, data):
        """Process incoming data and return results."""
        self.status = AgentStatus.RUNNING
        self.last_heartbeat = datetime.now()
        
        # Simulate processing time
        time.sleep(0.5)
        
        # Random chance of error for testing recovery
        if random.random() < 0.05:
            self.status = AgentStatus.ERROR
            logger.warning(f"Agent {self.agent_id} encountered an error")
            return {"error": "Simulated error", "agent_id": self.agent_id}
        
        # Return type-specific results
        if self.agent_type == AgentType.TECHNICAL_ANALYSIS:
            signals = self._generate_technical_signals(data)
            self.metrics["signal_count"] = len(signals)
            return {"signals": signals, "agent_id": self.agent_id}
            
        elif self.agent_type == AgentType.SENTIMENT_ANALYSIS:
            signals = self._generate_sentiment_signals(data)
            self.metrics["sentiment_intensity"] = sum(s["strength"] for s in signals) / len(signals) if signals else 0
            return {"signals": signals, "agent_id": self.agent_id}
            
        elif self.agent_type == AgentType.RISK_MANAGEMENT:
            constraints = self._generate_risk_constraints(data)
            self.metrics["risk_level"] = constraints["max_position_size"]
            return {"constraints": constraints, "agent_id": self.agent_id}
            
        elif self.agent_type == AgentType.DECISION:
            decisions = self._generate_decisions(data)
            self.metrics["decision_count"] = len(decisions)
            return {"decisions": decisions, "agent_id": self.agent_id}
            
        else:
            return {"result": "Generic processing", "agent_id": self.agent_id}
    
    def _generate_technical_signals(self, data):
        """Generate mock technical analysis signals."""
        signals = []
        for symbol in data.get("symbols", ["BTC/USD"]):
            signal_type = random.choice(["MA_CROSS", "RSI_OVERSOLD", "RSI_OVERBOUGHT", "MACD_CROSS"])
            strength = random.uniform(0.3, 0.9)
            direction = random.choice([1, -1])  # 1 for buy, -1 for sell
            signals.append({
                "symbol": symbol,
                "type": signal_type,
                "strength": strength,
                "direction": direction
            })
        return signals
    
    def _generate_sentiment_signals(self, data):
        """Generate mock sentiment analysis signals."""
        signals = []
        for symbol in data.get("symbols", ["BTC/USD"]):
            sentiment_score = random.uniform(-0.8, 0.8)
            confidence = random.uniform(0.4, 0.9)
            signals.append({
                "symbol": symbol,
                "sentiment_score": sentiment_score,
                "confidence": confidence,
                "strength": abs(sentiment_score) * confidence,
                "direction": 1 if sentiment_score > 0 else -1
            })
        return signals
    
    def _generate_risk_constraints(self, data):
        """Generate mock risk management constraints."""
        portfolio_value = data.get("portfolio_value", 100000)
        market_volatility = data.get("volatility_regime", VolatilityRegimeType.MEDIUM.value)
        
        # Adjust risk based on volatility
        if market_volatility == VolatilityRegimeType.LOW.value:
            max_position_size = 0.2
            stop_loss_pct = 0.05
        elif market_volatility == VolatilityRegimeType.MEDIUM.value:
            max_position_size = 0.15
            stop_loss_pct = 0.07
        elif market_volatility == VolatilityRegimeType.HIGH.value:
            max_position_size = 0.1
            stop_loss_pct = 0.1
        else:  # EXTREME
            max_position_size = 0.05
            stop_loss_pct = 0.15
        
        return {
            "max_position_size": max_position_size,
            "stop_loss_pct": stop_loss_pct,
            "max_portfolio_risk": 0.02,
            "max_correlation": 0.7
        }
    
    def _generate_decisions(self, data):
        """Generate mock trading decisions based on signals and constraints."""
        decisions = []
        
        tech_signals = []
        for result in data.get("agent_results", []):
            if "signals" in result and result.get("agent_id", "").startswith("technical"):
                tech_signals.extend(result["signals"])
        
        sentiment_signals = []
        for result in data.get("agent_results", []):
            if "signals" in result and result.get("agent_id", "").startswith("sentiment"):
                sentiment_signals.extend(result["signals"])
        
        # Get risk constraints
        constraints = {}
        for result in data.get("agent_results", []):
            if "constraints" in result:
                constraints = result["constraints"]
                break
        
        # Default constraints if none found
        if not constraints:
            constraints = {"max_position_size": 0.1, "stop_loss_pct": 0.05}
        
        # Generate decisions by combining signals
        symbols = data.get("symbols", ["BTC/USD"])
        for symbol in symbols:
            # Get signals for this symbol
            symbol_tech = [s for s in tech_signals if s.get("symbol") == symbol]
            symbol_sentiment = [s for s in sentiment_signals if s.get("symbol") == symbol]
            
            if not symbol_tech and not symbol_sentiment:
                continue
            
            # Calculate combined signal
            tech_direction = sum(s.get("direction", 0) * s.get("strength", 0.5) for s in symbol_tech)
            sent_direction = sum(s.get("direction", 0) * s.get("strength", 0.5) for s in symbol_sentiment)
            
            # Weight technical more than sentiment (configurable)
            combined_signal = (tech_direction * 0.7 + sent_direction * 0.3) / (len(symbol_tech) * 0.7 + len(symbol_sentiment) * 0.3) if (len(symbol_tech) + len(symbol_sentiment)) > 0 else 0
            
            # Apply risk constraints
            if abs(combined_signal) > 0.3:  # Signal threshold
                position_size = constraints.get("max_position_size", 0.1) * min(abs(combined_signal) * 2, 1.0)
                decisions.append({
                    "symbol": symbol,
                    "action": "buy" if combined_signal > 0 else "sell",
                    "position_size": position_size,
                    "stop_loss_pct": constraints.get("stop_loss_pct", 0.05),
                    "signal_strength": abs(combined_signal)
                })
        
        return decisions
    
    def recover(self):
        """Try to recover the agent from error state."""
        logger.info(f"Attempting recovery for agent {self.agent_id}")
        
        # Simulate recovery attempt
        recovery_success = random.random() < 0.8  # 80% success rate
        
        if recovery_success:
            self.status = AgentStatus.RUNNING
            logger.info(f"Agent {self.agent_id} successfully recovered")
            return True
        else:
            logger.error(f"Recovery failed for agent {self.agent_id}")
            return False
    
    def send_heartbeat(self):
        """Update agent heartbeat timestamp."""
        self.last_heartbeat = datetime.now()

class HealthMonitor:
    """Monitors the health of all registered components."""
    
    def __init__(self):
        self.components = {}
        self.heartbeat_threshold = timedelta(seconds=5)
        self.alerts = []
        self.recovery_actions = []
        logger.info("Health monitoring system initialized")
    
    def register_component(self, component_id, component):
        """Register a component for health monitoring."""
        self.components[component_id] = component
        logger.info(f"Registered component for health monitoring: {component_id}")
    
    def check_health(self):
        """Check the health of all registered components."""
        current_time = datetime.now()
        status = HealthStatus.HEALTHY
        
        for component_id, component in self.components.items():
            # Check heartbeat
            if current_time - component.last_heartbeat > self.heartbeat_threshold:
                self.alerts.append({
                    "component_id": component_id,
                    "alert_type": "missed_heartbeat",
                    "timestamp": current_time,
                    "message": f"Component {component_id} missed heartbeat"
                })
                status = HealthStatus.WARNING
            
            # Check component status
            if component.status == AgentStatus.ERROR:
                self.alerts.append({
                    "component_id": component_id,
                    "alert_type": "component_error",
                    "timestamp": current_time,
                    "message": f"Component {component_id} in error state"
                })
                status = HealthStatus.ERROR
                
                # Attempt recovery
                recovery_action = {
                    "component_id": component_id,
                    "action_type": "automatic_recovery",
                    "timestamp": current_time,
                    "message": f"Attempting automatic recovery for {component_id}"
                }
                self.recovery_actions.append(recovery_action)
                
                recovery_result = component.recover()
                if recovery_result:
                    recovery_action["result"] = "success"
                else:
                    recovery_action["result"] = "failed"
                    status = HealthStatus.CRITICAL
        
        return {
            "status": status,
            "alerts": self.alerts[-5:],  # Return last 5 alerts
            "recovery_actions": self.recovery_actions[-5:],  # Return last 5 recovery actions
            "components": {cid: comp.status.value for cid, comp in self.components.items()}
        }

class MarketRegimeClassifier:
    """Classifies market regimes based on price data."""
    
    def __init__(self):
        self.regime_history = {}
        logger.info("Market Regime Classifier initialized")
    
    def detect_regime(self, symbol, price_data=None):
        """Detect the current market regime for a symbol."""
        # In a real implementation, this would analyze price data
        # For this demo, we'll use random regimes with some persistence
        
        # If we already have a regime, 70% chance to keep it
        if symbol in self.regime_history and random.random() < 0.7:
            current_regime = self.regime_history[symbol]["regime_type"]
            current_volatility = self.regime_history[symbol]["volatility_regime"]
        else:
            current_regime = random.choice(list(MarketRegimeType)).value
            current_volatility = random.choice(list(VolatilityRegimeType)).value
        
        # Store in history
        self.regime_history[symbol] = {
            "regime_type": current_regime,
            "volatility_regime": current_volatility,
            "timestamp": datetime.now(),
            "confidence": random.uniform(0.7, 0.95)
        }
        
        return self.regime_history[symbol]
    
    def get_regime_history(self, symbol, days=30):
        """Get the regime history for a symbol."""
        if symbol not in self.regime_history:
            return None
        
        return self.regime_history[symbol]

class LLMOversight:
    """Simulates LLM oversight for trading decisions."""
    
    def __init__(self):
        self.decisions_reviewed = 0
        self.actions_taken = {
            "approve": 0,
            "modify": 0,
            "reject": 0,
            "log": 0
        }
        logger.info("LLM Oversight System initialized")
    
    def review_decision(self, decision):
        """Review a trading decision."""
        self.decisions_reviewed += 1
        
        # Check for risky decisions that need oversight
        risk_level = "low"
        if decision.get("position_size", 0) > 0.15:
            risk_level = "high"
        elif decision.get("position_size", 0) > 0.1:
            risk_level = "medium"
        
        # Randomly choose oversight action based on risk
        if risk_level == "high":
            action = random.choices(
                [OversightAction.APPROVE, OversightAction.MODIFY, OversightAction.REJECT],
                weights=[0.2, 0.5, 0.3]
            )[0]
        elif risk_level == "medium":
            action = random.choices(
                [OversightAction.APPROVE, OversightAction.MODIFY, OversightAction.LOG],
                weights=[0.5, 0.3, 0.2]
            )[0]
        else:
            action = random.choices(
                [OversightAction.APPROVE, OversightAction.LOG],
                weights=[0.9, 0.1]
            )[0]
        
        # Record the action
        self.actions_taken[action.value] += 1
        
        result = {
            "action": action.value,
            "risk_level": risk_level,
            "confidence": random.uniform(0.85, 0.98)
        }
        
        # Add reason and potential modifications
        if action == OversightAction.MODIFY:
            result["reason"] = "Position size exceeds risk threshold for current market conditions"
            result["modified_decision"] = decision.copy()
            result["modified_decision"]["position_size"] = decision.get("position_size", 0) * 0.7
            result["modified_decision"]["stop_loss_pct"] = decision.get("stop_loss_pct", 0.05) * 1.2
        elif action == OversightAction.REJECT:
            result["reason"] = "Decision violates risk management policies"
        elif action == OversightAction.APPROVE:
            result["reason"] = "Decision aligns with trading strategy and risk parameters"
        else:
            result["reason"] = "Decision logged for monitoring"
        
        return result
    
    def get_metrics(self):
        """Get oversight metrics."""
        return {
            "decisions_reviewed": self.decisions_reviewed,
            "actions_taken": self.actions_taken,
            "approval_rate": self.actions_taken["approve"] / self.decisions_reviewed if self.decisions_reviewed > 0 else 0
        }

class AdaptiveTrader:
    """Main orchestrator for the autonomous trading system."""
    
    def __init__(self):
        self.agents = {}
        self.health_monitor = HealthMonitor()
        self.regime_classifier = MarketRegimeClassifier()
        self.llm_oversight = LLMOversight()
        self.portfolio = {
            "value": 100000.0,
            "positions": {}
        }
        self.symbols = ["BTC/USD", "ETH/USD", "AAPL", "MSFT"]
        self.cycle_count = 0
        logger.info("Adaptive Trading System initialized")
    
    def setup_agents(self):
        """Set up all required agents."""
        # Technical analysis agents
        self.agents["technical_1"] = MockAgent("technical_1", AgentType.TECHNICAL_ANALYSIS)
        self.agents["technical_2"] = MockAgent("technical_2", AgentType.TECHNICAL_ANALYSIS)
        
        # Sentiment analysis agent
        self.agents["sentiment_1"] = MockAgent("sentiment_1", AgentType.SENTIMENT_ANALYSIS)
        
        # Risk management agent
        self.agents["risk_1"] = MockAgent("risk_1", AgentType.RISK_MANAGEMENT)
        
        # Decision agent
        self.agents["decision_1"] = MockAgent("decision_1", AgentType.DECISION)
        
        # Register all agents for health monitoring
        for agent_id, agent in self.agents.items():
            self.health_monitor.register_component(agent_id, agent)
        
        logger.info(f"Set up {len(self.agents)} agents for autonomous trading")
    
    def run_cycle(self):
        """Run a single trading cycle."""
        self.cycle_count += 1
        logger.info(f"\n----- Trading Cycle {self.cycle_count} -----")
        
        # 1. Check system health
        health_check = self.health_monitor.check_health()
        if health_check["status"] in [HealthStatus.ERROR, HealthStatus.CRITICAL]:
            logger.warning(f"System health issues detected: {health_check['status']}")
            for alert in health_check["alerts"]:
                logger.warning(f"Health alert: {alert['message']}")
            
            # Show recovery actions
            for action in health_check["recovery_actions"]:
                logger.info(f"Recovery action: {action['message']} - Result: {action.get('result', 'pending')}")
        
        # 2. Detect market regime for each symbol
        regimes = {}
        for symbol in self.symbols:
            regime = self.regime_classifier.detect_regime(symbol)
            regimes[symbol] = regime
            logger.info(f"Market regime for {symbol}: {regime['regime_type']} (Volatility: {regime['volatility_regime']})")
        
        # 3. Run technical analysis agents
        logger.info("Running technical analysis agents...")
        tech_results = []
        for agent_id, agent in self.agents.items():
            if agent.agent_type == AgentType.TECHNICAL_ANALYSIS:
                result = agent.process({"symbols": self.symbols, "regimes": regimes})
                if "error" not in result:
                    tech_results.append(result)
        
        # 4. Run sentiment analysis agent
        logger.info("Running sentiment analysis agents...")
        sentiment_results = []
        for agent_id, agent in self.agents.items():
            if agent.agent_type == AgentType.SENTIMENT_ANALYSIS:
                result = agent.process({"symbols": self.symbols, "regimes": regimes})
                if "error" not in result:
                    sentiment_results.append(result)
        
        # 5. Run risk management agent
        logger.info("Running risk management agents...")
        risk_results = []
        for agent_id, agent in self.agents.items():
            if agent.agent_type == AgentType.RISK_MANAGEMENT:
                result = agent.process({
                    "symbols": self.symbols, 
                    "regimes": regimes,
                    "portfolio_value": self.portfolio["value"],
                    "volatility_regime": next(iter(regimes.values()), {"volatility_regime": "medium"})["volatility_regime"]
                })
                if "error" not in result:
                    risk_results.append(result)
        
        # 6. Run decision agent
        logger.info("Running decision agent...")
        decisions = []
        for agent_id, agent in self.agents.items():
            if agent.agent_type == AgentType.DECISION:
                agent_results = tech_results + sentiment_results + risk_results
                result = agent.process({
                    "symbols": self.symbols,
                    "regimes": regimes,
                    "agent_results": agent_results
                })
                if "error" not in result and "decisions" in result:
                    decisions.extend(result["decisions"])
        
        # 7. Apply LLM oversight to each decision
        if decisions:
            logger.info("Applying LLM oversight to trading decisions...")
            for i, decision in enumerate(decisions):
                oversight_result = self.llm_oversight.review_decision(decision)
                logger.info(f"Decision {i+1}: {decision['action']} {decision['symbol']} - Position size: {decision['position_size']:.2f}")
                logger.info(f"Oversight: {oversight_result['action']} - {oversight_result['reason']}")
                
                # Apply modifications if needed
                if oversight_result['action'] == OversightAction.MODIFY.value:
                    decisions[i] = oversight_result['modified_decision']
                    logger.info(f"Modified decision: Position size adjusted to {decisions[i]['position_size']:.2f}")
                
                # Remove rejected decisions
                if oversight_result['action'] == OversightAction.REJECT.value:
                    logger.info(f"Decision rejected by oversight system")
        else:
            logger.info("No trading decisions generated in this cycle")
        
        # 8. Display summary of the cycle
        logger.info("\n----- Cycle Summary -----")
        logger.info(f"Health Status: {health_check['status'].name}")
        logger.info(f"Dominant Regime: {self.get_dominant_regime(regimes)}")
        logger.info(f"Technical Signals: {sum(len(r.get('signals', [])) for r in tech_results)}")
        logger.info(f"Sentiment Signals: {sum(len(r.get('signals', [])) for r in sentiment_results)}")
        logger.info(f"Final Decisions: {len([d for d in decisions if d.get('action')])}")
        
        # Update portfolio position with mock execution
        if decisions:
            self.execute_decisions(decisions)
        
        return {
            "health_status": health_check["status"].name,
            "regimes": {s: r["regime_type"] for s, r in regimes.items()},
            "decisions": decisions,
            "portfolio": self.portfolio
        }
    
    def get_dominant_regime(self, regimes):
        """Get the dominant market regime across all symbols."""
        if not regimes:
            return "unknown"
            
        regime_counts = {}
        for symbol, regime in regimes.items():
            regime_type = regime["regime_type"]
            if regime_type not in regime_counts:
                regime_counts[regime_type] = 0
            regime_counts[regime_type] += 1
        
        dominant_regime = max(regime_counts.items(), key=lambda x: x[1])
        return dominant_regime[0]
    
    def execute_decisions(self, decisions):
        """Execute trading decisions (simulated)."""
        for decision in decisions:
            symbol = decision.get("symbol")
            action = decision.get("action")
            position_size = decision.get("position_size", 0)
            
            if action == "buy":
                # Calculate position value
                position_value = self.portfolio["value"] * position_size
                
                # Update portfolio
                if symbol not in self.portfolio["positions"]:
                    self.portfolio["positions"][symbol] = 0
                self.portfolio["positions"][symbol] += position_value
                logger.info(f"Executed BUY for {symbol}: ${position_value:.2f} ({position_size*100:.1f}% of portfolio)")
                
            elif action == "sell":
                # If we don't have a position, this is a short position
                position_value = self.portfolio["value"] * position_size
                
                if symbol in self.portfolio["positions"] and self.portfolio["positions"][symbol] > 0:
                    # Exit long position
                    sold_amount = min(self.portfolio["positions"][symbol], position_value)
                    self.portfolio["positions"][symbol] -= sold_amount
                    logger.info(f"Executed SELL for {symbol}: ${sold_amount:.2f}")
                    
                    # Remove position if zero
                    if self.portfolio["positions"][symbol] <= 0:
                        del self.portfolio["positions"][symbol]
                else:
                    # Short selling not implemented in this demo
                    logger.info(f"Short selling not implemented: Ignored SELL signal for {symbol}")
        
        # Update portfolio value with random market movement
        portfolio_change = random.uniform(-0.02, 0.03)  # -2% to +3% change
        self.portfolio["value"] *= (1 + portfolio_change)
        logger.info(f"Portfolio value updated: ${self.portfolio['value']:.2f} (Change: {portfolio_change*100:+.2f}%)")

def run_demo(cycles=5):
    """Run the autonomous trading system demo."""
    logger.info("Starting AI Trading Agent Autonomous Demo")
    
    # Create the adaptive trader
    trader = AdaptiveTrader()
    
    # Set up agents
    trader.setup_agents()
    
    # Run trading cycles
    for _ in range(cycles):
        trader.run_cycle()
        
        # Pause between cycles
        time.sleep(1)
    
    # Show final results
    logger.info("\n===== Demo Complete =====")
    logger.info(f"Completed {cycles} autonomous trading cycles")
    logger.info(f"Final portfolio value: ${trader.portfolio['value']:.2f}")
    logger.info(f"Positions: {trader.portfolio['positions']}")
    
    # Show oversight metrics
    oversight_metrics = trader.llm_oversight.get_metrics()
    logger.info("\nLLM Oversight Metrics:")
    logger.info(f"Decisions reviewed: {oversight_metrics['decisions_reviewed']}")
    logger.info(f"Approval rate: {oversight_metrics['approval_rate']*100:.1f}%")
    for action, count in oversight_metrics['actions_taken'].items():
        logger.info(f"{action.capitalize()}: {count}")

if __name__ == "__main__":
    run_demo(cycles=10)
