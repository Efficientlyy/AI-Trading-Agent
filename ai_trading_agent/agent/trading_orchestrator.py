from typing import Dict, List, Optional, Any, Tuple
from collections import deque
import logging
import time
import os
import json
from datetime import datetime
import threading

# Assuming agent_definitions.py is in the same directory or accessible via PYTHONPATH
try:
    from .agent_definitions import BaseAgent, AgentStatus
    from .recovery_manager import RecoveryManager, RecoveryState, RecoveryStrategy
except ImportError:
    # Fallback for environments where the relative import might not work directly (e.g. some test runners)
    from agent_definitions import BaseAgent, AgentStatus
    from recovery_manager import RecoveryManager, RecoveryState, RecoveryStrategy

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TradingOrchestrator:
    """
    Manages a collection of agents and orchestrates the flow of data and signals
    between them based on their defined inputs and outputs.
    
    Features:
    - Dynamic agent registration and dependency resolution
    - Automated execution order determination
    - Data routing between agents
    - Fault detection and recovery
    - Transaction journaling and recovery
    - Agent state preservation and restoration
    """
    def __init__(self, recovery_data_dir: str = "./data/recovery"):
        self.agents: Dict[str, BaseAgent] = {}
        self.agent_execution_order: List[str] = [] # To be determined by dependencies
        self.data_queues: Dict[str, deque] = {} # agent_id -> deque of pending data/signals
        
        # Fault tolerance and recovery
        self.recovery_manager = RecoveryManager(data_dir=recovery_data_dir)
        self.agent_health: Dict[str, Dict[str, Any]] = {} # agent_id -> health metrics
        self.last_checkpoint_time: Dict[str, float] = {} # agent_id -> last checkpoint time
        self.checkpoint_interval = 60  # seconds
        
        # Transaction tracking
        self.transactions: Dict[str, Dict[str, Any]] = {} # transaction_id -> transaction info
        
        # Background monitoring thread
        self.monitoring_thread = None
        self.monitoring_active = False
        self.monitoring_interval = 10  # seconds

    def register_agent(self, agent: BaseAgent):
        """Registers an agent with the orchestrator."""
        if agent.agent_id in self.agents:
            logger.warning(f"Agent with ID {agent.agent_id} already registered. Overwriting.")
        self.agents[agent.agent_id] = agent
        self.data_queues[agent.agent_id] = deque()
        
        # Initialize health tracking
        self.agent_health[agent.agent_id] = {
            "status": "initialized",
            "last_successful_run": None,
            "error_count": 0,
            "consecutive_errors": 0,
            "last_error": None,
            "recovery_attempts": 0
        }
        
        # Initialize checkpoint tracking
        self.last_checkpoint_time[agent.agent_id] = time.time()
        
        logger.info(f"Agent {agent.agent_id} ({agent.name}) registered.")
        self._determine_execution_order()

    def _determine_execution_order(self):
        """
        Determines a valid execution order for agents based on dependencies
        using Kahn's algorithm for topological sorting.
        This helps in processing agents in an order where their data dependencies are met.
        It also detects circular dependencies.
        """
        if not self.agents:
            self.agent_execution_order = []
            print("No agents registered. Execution order is empty.")
            return

        # Graph representation: adjacency list (agent_id -> list of agents it outputs to)
        adj: Dict[str, List[str]] = {agent_id: [] for agent_id in self.agents}
        # In-degree: number of incoming edges (dependencies) for each agent
        in_degree: Dict[str, int] = {agent_id: 0 for agent_id in self.agents}

        for agent_id, agent_instance in self.agents.items():
            # For each agent, look at who it receives input FROM (its dependencies)
            # If agent A receives input from agent B, then B is a prerequisite for A.
            # So, an edge goes from B to A.
            for source_agent_id in agent_instance.inputs_from:
                if source_agent_id in self.agents: # Ensure the source agent is registered
                    adj[source_agent_id].append(agent_id)
                    in_degree[agent_id] += 1
                else:
                    print(f"Warning: Agent {agent_id} lists unregistered agent {source_agent_id} as an input. Ignoring this dependency.")
        
        # Initialize a queue with all agents having an in-degree of 0 (no prerequisites)
        queue = deque([agent_id for agent_id, degree in in_degree.items() if degree == 0])
        
        sorted_order: List[str] = []
        
        while queue:
            current_agent_id = queue.popleft()
            sorted_order.append(current_agent_id)
            
            # For each neighbor (agent that depends on current_agent_id)
            for neighbor_agent_id in adj.get(current_agent_id, []):
                in_degree[neighbor_agent_id] -= 1
                if in_degree[neighbor_agent_id] == 0:
                    queue.append(neighbor_agent_id)
                    
        if len(sorted_order) == len(self.agents):
            self.agent_execution_order = sorted_order
            print(f"Determined agent execution order: {self.agent_execution_order}")
        else:
            # If sorted_order is shorter, there's a cycle
            self.agent_execution_order = [] # Or could be partially filled, but an error state is better
            
            # Identify agents involved in the cycle (those with in_degree > 0)
            # This is a simplified way to show problematic agents; more complex cycle detection might be needed for exact cycle paths.
            remaining_agents = [agent_id for agent_id, degree in in_degree.items() if degree > 0]
            print(f"Error: Circular dependency detected in agent graph. Could not determine execution order. Problematic agents (or part of cycle): {remaining_agents}")
            # Fallback: use registration order or alphabetical, but this is not ideal
            # For safety, set to empty and let run_cycle fail if no order.
            # self.agent_execution_order = list(self.agents.keys())
            # print(f"Warning: Using fallback execution order (registration order) due to cycle: {self.agent_execution_order}")


    def get_agent(self, agent_id: str) -> Optional[BaseAgent]:
        """Retrieves a registered agent by its ID."""
        return self.agents.get(agent_id)

    def start_all_agents(self):
        """Starts all registered agents."""
        for agent_id, agent in self.agents.items():
            try:
                agent.start()
                
                # Update agent health status
                self.agent_health[agent_id]["status"] = "running"
                
                logger.info(f"Started agent {agent_id} ({agent.name})")
            except Exception as e:
                logger.error(f"Failed to start agent {agent_id}: {str(e)}")
                
                # Record failure in health tracking
                self.agent_health[agent_id]["status"] = "failed"
                self.agent_health[agent_id]["last_error"] = {
                    "time": datetime.now().isoformat(),
                    "error": str(e),
                    "phase": "startup"
                }
                self.agent_health[agent_id]["error_count"] += 1
                self.agent_health[agent_id]["consecutive_errors"] += 1
                
                # Attempt recovery
                self._attempt_agent_recovery(agent_id, {
                    "type": "startup_failure",
                    "severity": "high",
                    "reason": str(e)
                })
        
        # Start monitoring if not already running
        self._start_monitoring()

    def stop_all_agents(self):
        """Stops all registered agents."""
        print("Stopping all registered agents...")
        # Stop in reverse order of execution if necessary, or just iterate
        for agent_id in reversed(self.agent_execution_order):
            agent = self.agents.get(agent_id)
            if agent:
                try:
                    agent.stop()
                except Exception as e:
                    print(f"Error stopping agent {agent.agent_id}: {e}")
                    # Status might already be STOPPED or ERROR

    def route_output(self, producing_agent_id: str, output_data: Optional[any]): # Output can be Dict or List[Dict]
        """
        Routes the output of an agent to its designated recipients.
        Handles both single dictionary outputs and lists of dictionaries.
        """
        if not output_data:
            return

        producer = self.agents.get(producing_agent_id)
        if not producer:
            print(f"Error: Producer agent {producing_agent_id} not found for routing.")
            return

        outputs_to_route = []
        if isinstance(output_data, list):
            outputs_to_route.extend(output_data)
        elif isinstance(output_data, dict):
            outputs_to_route.append(output_data)
        else:
            print(f"Warning: Agent {producing_agent_id} produced output of unexpected type: {type(output_data)}. Expected Dict or List[Dict].")
            return

        for item in outputs_to_route:
            if not isinstance(item, dict):
                print(f"Warning: Agent {producing_agent_id} produced a list containing a non-dictionary item: {type(item)}. Skipping this item.")
                continue
            for recipient_id in producer.outputs_to:
                if recipient_id in self.data_queues:
                    self.data_queues[recipient_id].append(item)
                    # print(f"Routed output from {producing_agent_id} to {recipient_id}: {item}")
                else:
                    print(f"Warning: Recipient agent {recipient_id} (from {producing_agent_id}) not found or has no queue for item: {item}.")

    def run_cycle(self, external_inputs: Optional[Dict[str, List[Dict]]] = None):
        """
        Runs one cycle of processing for all agents in the determined order.
        External inputs are keyed by agent_id.
        """
        if not self.agent_execution_order:
            logger.warning("No agent execution order determined. Cannot run processing cycle.")
            return
        
        # Process external inputs
        if external_inputs:
            for agent_id, inputs in external_inputs.items():
                if agent_id in self.data_queues:
                    for input_item in inputs:
                        self.data_queues[agent_id].append(input_item)
                else:
                    logger.warning(f"External input for unknown agent {agent_id}. Ignoring.")
        
        # Check for and reconcile any orphaned transactions
        self.recovery_manager.reconcile_orphaned_transactions()
        
        # Process each agent in the execution order
        for agent_id in self.agent_execution_order:
            agent = self.agents[agent_id]
            
            # Early skip if agent is not active
            if agent.status != AgentStatus.ACTIVE:
                logger.info(f"Agent {agent_id} is not active (status: {agent.status}). Skipping.")
                continue
            
            # Get any queued data for this agent
            inputs = []
            if self.data_queues[agent_id]:
                while self.data_queues[agent_id]:
                    inputs.append(self.data_queues[agent_id].popleft())
                
                logger.debug(f"Agent {agent_id} processing {len(inputs)} queued inputs")
                
                # Process the inputs for this agent
                try:
                    cycle_start_time = time.time()
                    
                    # Checkpoint state if needed
                    self._maybe_checkpoint_agent_state(agent_id)
                    
                    # Process inputs
                    output = agent.process(inputs)
                    
                    # Record successful processing
                    cycle_end_time = time.time()
                    process_time_ms = (cycle_end_time - cycle_start_time) * 1000
                    
                    # Update health metrics
                    self.agent_health[agent_id]["last_successful_run"] = datetime.now().isoformat()
                    self.agent_health[agent_id]["consecutive_errors"] = 0
                    self.agent_health[agent_id]["status"] = "running"
                    self.agent_health[agent_id]["last_process_time_ms"] = process_time_ms
                    
                    if output is not None:
                        # Route the output to other agents that depend on this one
                        self.route_output(agent_id, output)
                        
                except Exception as e:
                    logger.error(f"Error processing agent {agent_id}: {str(e)}")
                    
                    # Update health metrics
                    self.agent_health[agent_id]["error_count"] += 1
                    self.agent_health[agent_id]["consecutive_errors"] += 1
                    self.agent_health[agent_id]["last_error"] = {
                        "time": datetime.now().isoformat(),
                        "error": str(e),
                        "phase": "processing"
                    }
                    
                    # Attempt recovery if needed
                    if self.agent_health[agent_id]["consecutive_errors"] >= 3:
                        logger.warning(f"Agent {agent_id} has failed {self.agent_health[agent_id]['consecutive_errors']} consecutive times. Attempting recovery.")
                        self._attempt_agent_recovery(agent_id, {
                            "type": "consecutive_failures",
                            "severity": "medium",
                            "reason": str(e)
                        })
            else:
                # No inputs for this agent, still allow it to run if needed (e.g., for agents that poll or generate data)
                logger.debug(f"Agent {agent_id} has no queued inputs, but running process cycle")
                try:
                    cycle_start_time = time.time()
                    
                    # Checkpoint state if needed
                    self._maybe_checkpoint_agent_state(agent_id)
                    
                    # Process
                    output = agent.process([])
                    
                    # Record successful processing
                    cycle_end_time = time.time()
                    process_time_ms = (cycle_end_time - cycle_start_time) * 1000
                    
                    # Update health metrics
                    self.agent_health[agent_id]["last_successful_run"] = datetime.now().isoformat()
                    self.agent_health[agent_id]["consecutive_errors"] = 0
                    self.agent_health[agent_id]["status"] = "running"
                    self.agent_health[agent_id]["last_process_time_ms"] = process_time_ms
                    
                    if output is not None:
                        # Route the output to other agents that depend on this one
                        self.route_output(agent_id, output)
                except Exception as e:
                    logger.error(f"Error processing agent {agent_id}: {str(e)}")
                    
                    # Update health metrics
                    self.agent_health[agent_id]["error_count"] += 1
                    self.agent_health[agent_id]["consecutive_errors"] += 1
                    self.agent_health[agent_id]["last_error"] = {
                        "time": datetime.now().isoformat(),
                        "error": str(e),
                        "phase": "processing"
                    }
                    
                    # Attempt recovery if needed
                    if self.agent_health[agent_id]["consecutive_errors"] >= 3:
                        logger.warning(f"Agent {agent_id} has failed {self.agent_health[agent_id]['consecutive_errors']} consecutive times. Attempting recovery.")
                        self._attempt_agent_recovery(agent_id, {
                            "type": "consecutive_failures",
                            "severity": "medium",
                            "reason": str(e)
                        })

    def get_all_agent_info(self):
        """Returns a list of info dictionaries for all registered agents."""
        agent_info = []
        for agent_id, agent in self.agents.items():
            # Get basic agent info
            info = agent.get_info()
            
            # Add health information
            if agent_id in self.agent_health:
                info["health"] = self.agent_health[agent_id]
            
            agent_info.append(info)
            
        return agent_info
        
    def _start_monitoring(self):
        """Start the background monitoring thread."""
        if self.monitoring_thread is not None and self.monitoring_active:
            logger.info("Monitoring thread already running")
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        logger.info("Started agent monitoring thread")
    
    def _stop_monitoring(self):
        """Stop the background monitoring thread."""
        self.monitoring_active = False
        if self.monitoring_thread is not None:
            self.monitoring_thread.join(timeout=2.0)
            self.monitoring_thread = None
            logger.info("Stopped agent monitoring thread")
    
    def _monitoring_loop(self):
        """Background thread for monitoring agent health."""
        while self.monitoring_active:
            try:
                # Check each agent's health
                for agent_id, agent in self.agents.items():
                    if agent.status == AgentStatus.ACTIVE:
                        # Check if agent is responsive
                        if not self._is_agent_healthy(agent_id):
                            logger.warning(f"Agent {agent_id} appears to be unresponsive")
                            
                            # Update health metrics
                            self.agent_health[agent_id]["status"] = "unresponsive"
                            
                            # Attempt recovery
                            self._attempt_agent_recovery(agent_id, {
                                "type": "unresponsive",
                                "severity": "high",
                                "reason": "Agent is not responding to health checks"
                            })
                
                # Sleep for monitoring interval
                time.sleep(self.monitoring_interval)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {str(e)}")
                time.sleep(5)  # Sleep briefly on error
    
    def _is_agent_healthy(self, agent_id: str) -> bool:
        """Check if an agent is responsive and healthy."""
        agent = self.agents.get(agent_id)
        if not agent:
            return False
        
        # If agent has a health_check method, use it
        if hasattr(agent, 'health_check') and callable(getattr(agent, 'health_check')):
            try:
                result = agent.health_check()
                return result.get('healthy', False)
            except Exception as e:
                logger.error(f"Error in health check for agent {agent_id}: {str(e)}")
                return False
        
        # Otherwise use basic checks
        if agent.status != AgentStatus.ACTIVE:
            return False
        
        # Check last successful run (if any)
        health = self.agent_health.get(agent_id, {})
        last_run = health.get("last_successful_run")
        if not last_run:
            # New agent, no runs yet
            return True
        
        # Check if agent has run recently - timeout based on agent type
        # Different agent types might have different expected cycle times
        max_inactive_time = 300  # Default: 5 minutes
        
        # Agent-specific timeouts could be defined here
        # For example, real-time agents might have shorter timeouts
        
        try:
            last_run_time = datetime.fromisoformat(last_run)
            seconds_since_last_run = (datetime.now() - last_run_time).total_seconds()
            return seconds_since_last_run < max_inactive_time
        except Exception:
            return False
    
    def _attempt_agent_recovery(self, agent_id: str, failure_info: Dict[str, Any]) -> bool:
        """Attempt to recover a failed agent."""
        logger.info(f"Attempting to recover agent {agent_id}: {failure_info.get('reason', 'Unknown failure')}")
        
        # Update recovery attempt count
        self.agent_health[agent_id]["recovery_attempts"] += 1
        
        # Get agent instance
        agent = self.agents.get(agent_id)
        if not agent:
            logger.error(f"Cannot recover unknown agent {agent_id}")
            return False
        
        # Use recovery manager to determine and apply recovery strategy
        success, strategy = self.recovery_manager.handle_agent_failure(agent_id, failure_info)
        if not success:
            logger.error(f"Failed to determine recovery strategy for agent {agent_id}")
            return False
        
        # Apply recovery strategy
        success = self.recovery_manager.apply_recovery_strategy(agent_id, strategy, agent)
        
        # Update health status based on recovery result
        if success:
            logger.info(f"Successfully recovered agent {agent_id} using strategy {strategy.value}")
            self.agent_health[agent_id]["status"] = "recovered"
            self.agent_health[agent_id]["consecutive_errors"] = 0
        else:
            logger.error(f"Failed to recover agent {agent_id} using strategy {strategy.value}")
            self.agent_health[agent_id]["status"] = "failed"
        
        return success
    
    def _checkpoint_agent_state(self, agent_id: str) -> bool:
        """Checkpoint agent state for potential recovery."""
        agent = self.agents.get(agent_id)
        if not agent:
            logger.warning(f"Cannot checkpoint unknown agent {agent_id}")
            return False
        
        # Get agent state
        state = None
        if hasattr(agent, 'get_state') and callable(getattr(agent, 'get_state')):
            try:
                state = agent.get_state()
            except Exception as e:
                logger.error(f"Error getting state for agent {agent_id}: {str(e)}")
                return False
        else:
            # Create basic state from agent attributes
            state = {}
            # Safely copy serializable attributes
            for attr in dir(agent):
                if not attr.startswith('_') and not callable(getattr(agent, attr)):
                    try:
                        value = getattr(agent, attr)
                        # Try to serialize - only include serializable values
                        json.dumps({attr: value})
                        state[attr] = value
                    except (TypeError, OverflowError, ValueError):
                        # Skip non-serializable attributes
                        pass
        
        # Save state using recovery manager
        if state:
            success = self.recovery_manager.checkpoint_agent_state(agent_id, state)
            if success:
                self.last_checkpoint_time[agent_id] = time.time()
                logger.debug(f"Checkpointed state for agent {agent_id}")
            return success
        else:
            logger.warning(f"No state to checkpoint for agent {agent_id}")
            return False
    
    def _maybe_checkpoint_agent_state(self, agent_id: str) -> None:
        """Checkpoint agent state if enough time has passed since last checkpoint."""
        last_time = self.last_checkpoint_time.get(agent_id, 0)
        current_time = time.time()
        
        if current_time - last_time >= self.checkpoint_interval:
            self._checkpoint_agent_state(agent_id)
    
    def journal_transaction(self, transaction_type: str, transaction_data: Dict[str, Any], agent_id: str) -> str:
        """Journal a transaction for recovery purposes."""
        # Use recovery manager to journal the transaction
        transaction_id = self.recovery_manager.journal_transaction(transaction_type, transaction_data, agent_id)
        
        # Track transaction locally
        self.transactions[transaction_id] = {
            "type": transaction_type,
            "agent_id": agent_id,
            "data": transaction_data,
            "status": "pending",
            "timestamp": datetime.now().isoformat()
        }
        
        return transaction_id
    
    def update_transaction_status(self, transaction_id: str, status: str, result: Optional[Dict[str, Any]] = None) -> bool:
        """Update the status of a journaled transaction."""
        if transaction_id not in self.transactions:
            logger.warning(f"Unknown transaction ID: {transaction_id}")
            return False
        
        # Update local tracking
        self.transactions[transaction_id]["status"] = status
        if result:
            self.transactions[transaction_id]["result"] = result
        self.transactions[transaction_id]["updated_at"] = datetime.now().isoformat()
        
        # Update in recovery manager
        return self.recovery_manager.update_transaction_status(transaction_id, status, result)
    
    def get_recovery_status(self) -> Dict[str, Any]:
        """Get current recovery status information."""
        return self.recovery_manager.get_recovery_status()
    
    def get_agent_health_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get health metrics for all agents."""
        return self.agent_health.copy()

if __name__ == '__main__':
    # Import example agents for testing
    from agent_definitions import (
        SentimentAnalysisAgent, TechnicalAnalysisAgent, NewsEventAgent, FundamentalAnalysisAgent,
        DecisionAgent, ExecutionLayerAgent, AgentRole, AgentStatus
    )

    orchestrator = TradingOrchestrator()

    # Instantiate agents
    # Sentiment Agents
    sentiment_agent_btc = SentimentAnalysisAgent(
        agent_id_suffix="av_news_btc", name="AV BTC News Sentiment",
        agent_type="AlphaVantageNews", symbols=["BTC/USD"],
        config_details={"api_key": "YOUR_AV_KEY_HERE"}
    )
    sentiment_agent_eth = SentimentAnalysisAgent(
        agent_id_suffix="av_news_eth", name="AV ETH News Sentiment",
        agent_type="AlphaVantageNews", symbols=["ETH/USD"],
        config_details={"api_key": "YOUR_AV_KEY_HERE"}
    )

    # Technical Agents
    technical_agent_btc = TechnicalAnalysisAgent(
        agent_id_suffix="ta_btc", name="TA BTC",
        agent_type="RSIMACDStrategy", symbols=["BTC/USD"],
        config_details={"rsi_period": 14, "macd_fast": 12}
    )
    technical_agent_eth = TechnicalAnalysisAgent(
        agent_id_suffix="ta_eth", name="TA ETH",
        agent_type="RSIMACDStrategy", symbols=["ETH/USD"]
    )

    # News Event Agent
    news_event_agent_global = NewsEventAgent(
        agent_id_suffix="global_events", name="Global News Event Monitor",
        agent_type="GeneralEventScanner",
        symbols=["BTC/USD", "ETH/USD", "X", "PHRM"], # Monitors multiple symbols for events
        event_keywords=["earnings", "fda approval", "regulatory change", "guidance", "partnership"]
    )
    
    # Fundamental Analysis Agent
    fundamental_agent_x = FundamentalAnalysisAgent(
        agent_id_suffix="stock_x_fundamentals", name="Stock X Fundamentals",
        agent_type="CompanyValuator", symbols=["X"] # Focuses on specific stock "X"
    )
    fundamental_agent_phrm = FundamentalAnalysisAgent(
        agent_id_suffix="stock_phrm_fundamentals", name="Stock PHRM Fundamentals",
        agent_type="CompanyValuator", symbols=["PHRM"]
    )

    # Decision Agent
    decision_agent_main = DecisionAgent(
        agent_id_suffix="main_v1", name="Main Decision Logic V1",
        agent_type="WeightedSignalAggregator",
        config_details={
            "min_signals_for_decision": 2,
            "buy_threshold": 0.55,
            "sell_threshold": -0.45,
            "signal_weights": {
                "sentiment_signal": 0.3,
                "technical_signal": 0.35,
                "news_event_signal": 0.25,
                "fundamental_signal": 0.1
            },
            "risk_management": { # Added risk management configuration
                "default_trade_quantity": 0.05,
                "max_trade_value_usd": 150.0,
                "per_symbol_max_quantity": {
                    "BTC/USD": 0.1,
                    "ETH/USD": 0.5,
                    "X": 10,
                    "PHRM": 20
                }
            }
        }
    )
    
    # Execution Agent
    execution_agent_paper = ExecutionLayerAgent(
        agent_id_suffix="paper_trader", name="Paper Trading Executor",
        agent_type="InternalPaperBroker"
    )

    # Define connections: All specialized agents output to the main decision agent
    sentiment_agent_btc.outputs_to = [decision_agent_main.agent_id]
    sentiment_agent_eth.outputs_to = [decision_agent_main.agent_id]
    technical_agent_btc.outputs_to = [decision_agent_main.agent_id]
    technical_agent_eth.outputs_to = [decision_agent_main.agent_id]
    news_event_agent_global.outputs_to = [decision_agent_main.agent_id]
    fundamental_agent_x.outputs_to = [decision_agent_main.agent_id]
    fundamental_agent_phrm.outputs_to = [decision_agent_main.agent_id]

    decision_agent_main.inputs_from = [
        sentiment_agent_btc.agent_id, sentiment_agent_eth.agent_id,
        technical_agent_btc.agent_id, technical_agent_eth.agent_id,
        news_event_agent_global.agent_id,
        fundamental_agent_x.agent_id, fundamental_agent_phrm.agent_id
    ]
    decision_agent_main.outputs_to = [execution_agent_paper.agent_id]
    execution_agent_paper.inputs_from = [decision_agent_main.agent_id]

    # Register all agents
    orchestrator.register_agent(sentiment_agent_btc)
    orchestrator.register_agent(sentiment_agent_eth)
    orchestrator.register_agent(technical_agent_btc)
    orchestrator.register_agent(technical_agent_eth)
    orchestrator.register_agent(news_event_agent_global)
    orchestrator.register_agent(fundamental_agent_x)
    orchestrator.register_agent(fundamental_agent_phrm)
    orchestrator.register_agent(decision_agent_main)
    orchestrator.register_agent(execution_agent_paper)
    
    print("\n--- Orchestrator Initialized ---")
    print(f"Execution Order: {orchestrator.agent_execution_order}")

    orchestrator.start_all_agents()

    print("\n--- Simulating Trading Cycle 1 (Agents fetch their own data) ---")
    orchestrator.run_cycle()
    # At this point, specialized agents should have processed and their outputs (signals)
    # should be in the decision_agent_main's queue.

    print("\n--- Simulating Trading Cycle 2 (Decision agent processes queued signals) ---")
    orchestrator.run_cycle()
    # decision_agent_main processes signals from cycle 1. If a directive is created,
    # it's queued for execution_agent_paper.

    print("\n--- Simulating Trading Cycle 3 (Execution agent processes directive if any) ---")
    orchestrator.run_cycle()
    # execution_agent_paper processes directive from cycle 2. Feedback might be generated.

    print("\n--- Simulating Trading Cycle 4 (Optional: Feedback processing or new external data) ---")
    # Example: Manually push a query to the execution agent for a position
    # This would typically originate from a user interface or another management agent.
    if execution_agent_paper.agent_id in orchestrator.data_queues:
         print(f"\n--- Manually queueing position query for {execution_agent_paper.agent_id} ---")
         orchestrator.data_queues[execution_agent_paper.agent_id].append(
             {"type": "query_position", "payload": {"symbol": "BTC/USD"}}
         )
    orchestrator.run_cycle() # Execution agent processes query, its output is routed.
    
    # If execution agent sent feedback to decision agent (if configured), one more cycle for that.
    # For now, feedback is not explicitly routed back in this example setup.

    print("\n--- All Agent Info from Orchestrator ---")
    all_info = orchestrator.get_all_agent_info()
    for info in all_info:
        print(info)

    orchestrator.stop_all_agents()
    print("\n--- Orchestration Test Complete ---")
