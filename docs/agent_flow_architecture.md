# AI Trading Agent - Agent Flow Architecture

## 1. Introduction

This document outlines the architecture of the multi-agent system within the AI Trading Agent platform. It describes the different roles of agents, how they interact, and the flow of data and signals through the system. This serves as a blueprint for visualizing the agent interactions on the System Control Dashboard.

The system is designed around three core types of agents: Specialized Agents, Decision Agent(s), and an Execution Layer Agent.

## 2. Agent Roles and Responsibilities

### 2.1. Specialized Agents

Specialized agents are responsible for analyzing unique data streams or applying specific trading strategies to generate primary trading signals or analytical insights. Each specialized agent focuses on a particular domain.

*   **Sentiment Analysis Agent:**
    *   **Data Sources:** News APIs (e.g., Alpha Vantage News & Sentiment), social media feeds, financial forums.
    *   **Responsibilities:** Processes textual data, performs sentiment scoring, identifies sentiment trends, and generates sentiment-based signals (e.g., bullish/bearish sentiment for an asset).
    *   **Example `agent_id` prefix:** `spec_sentiment_`
*   **Technical Analysis (Chart) Agent:**
    *   **Data Sources:** Real-time market data providers (e.g., Twelve Data), historical price data.
    *   **Responsibilities:** Calculates various technical indicators (e.g., MA, RSI, MACD, Bollinger Bands), identifies chart patterns, and generates signals based on technical criteria (e.g., crossover events, breakout confirmations).
    *   **Example `agent_id` prefix:** `spec_technical_`
*   **News Event Agent (Future):**
    *   **Data Sources:** Financial news APIs, press release aggregators.
    *   **Responsibilities:** Identifies significant market-moving news events (e.g., earnings reports, regulatory changes, macroeconomic announcements) and generates event-based signals.
    *   **Example `agent_id` prefix:** `spec_news_`
*   **Fundamental Analysis Agent (Future):**
    *   **Data Sources:** Company financial statements, economic data providers.
    *   **Responsibilities:** Analyzes fundamental data (e.g., P/E ratios, revenue growth, debt levels) to assess asset valuation and generate long-term investment signals.
    *   **Example `agent_id` prefix:** `spec_fundamental_`

### 2.2. Decision Agent(s)

The Decision Agent acts as the central brain of the trading system. It aggregates signals and insights from all active Specialized Agents, applies overarching logic, and makes the final trading decisions.

*   **Responsibilities:**
    *   Collects and normalizes signals from various Specialized Agents.
    *   Applies a configurable weighting or scoring system to different signals.
    *   Incorporates risk management rules (e.g., max drawdown, position sizing limits).
    *   Considers portfolio allocation constraints and diversification goals.
    *   Determines the final trading action (buy, sell, hold), quantity, and order type.
    *   May involve a hierarchy or multiple decision agents for different strategies or asset classes.
*   **Example `agent_id` prefix:** `decision_`

### 2.3. Execution Layer Agent

The Execution Layer Agent is responsible for interacting with the brokerage or exchange APIs to carry out the trading decisions made by the Decision Agent.

*   **Responsibilities:**
    *   Receives trading directives (e.g., buy 1 BTC at market price).
    *   Places orders with the broker/exchange.
    *   Monitors order status (filled, partially filled, canceled).
    *   Manages open positions.
    *   Provides execution feedback (e.g., fill price, fees) back to the system for performance tracking and potential adaptation by other agents.
*   **Example `agent_id` prefix:** `exec_`

## 3. Data and Signal Flow Diagram

The following diagram illustrates the typical flow of information between the agent types:

```mermaid
graph TD
    A[Data Sources (Market Data, News, Sentiment Feeds)] --> SA1;
    A --> SA2;
    A --> SAN;

    subgraph Specialized Agents
        SA1[Sentiment Analysis Agent]
        SA2[Technical Analysis Agent]
        SAN[... Other Specialized Agents ...]
    end

    subgraph Decision Logic
        DA[Decision Agent]
    end

    subgraph Broker/Exchange Interface
        EX[Execution Layer Agent]
    end

    SA1 -- Sentiment Signals --> DA;
    SA2 -- Technical Signals --> DA;
    SAN -- Other Signals --> DA;
    
    DA -- Trading Directives (Buy/Sell/Hold, Size) --> EX;
    EX -- Execution Status & Feedback --> DA;
    EX -- Execution Status & Feedback --> PM[Portfolio Management / Monitoring];
    
    style SA1 fill:#D5E8D4,stroke:#82B366,stroke-width:2px
    style SA2 fill:#D5E8D4,stroke:#82B366,stroke-width:2px
    style SAN fill:#D5E8D4,stroke:#82B366,stroke-width:2px
    style DA fill:#DAE8FC,stroke:#6C8EBF,stroke-width:2px
    style EX fill:#FFE6CC,stroke:#D79B00,stroke-width:2px
    style A fill:#f9f9f9,stroke:#333,stroke-width:2px
    style PM fill:#E1D5E7,stroke:#9673A6,stroke-width:2px

```

## 4. Conceptual Data Structure for Agents

To support this architecture and its visualization, agent data should include fields like:

*   `agent_id`: (string) Unique identifier.
*   `name`: (string) Display name for the agent.
*   `agent_role`: (string) Enum or defined string indicating the role (e.g., `specialized_sentiment`, `specialized_technical`, `decision_aggregator`, `execution_broker`).
*   `type`: (string) More specific type or strategy name (e.g., "AlphaVantageSentiment", "RSIMACDStrategy", "MainDecisionLogicV1", "AlpacaBroker").
*   `status`: (string) Current operational status (e.g., `running`, `stopped`, `error`, `initializing`).
*   `inputs_from`: (Optional[List[string]]) List of `agent_id`s from which this agent receives primary data or signals.
*   `outputs_to`: (Optional[List[string]]) List of `agent_id`s to which this agent sends its primary output or signals.
*   `config_details`: (Optional[Dict]) Key configuration parameters specific to the agent.
*   `metrics`: (Optional[Dict]) Performance or operational metrics.
*   `last_updated`: (datetime) Timestamp of the last status or data update.
*   `symbols`: (Optional[List[string]]) For specialized agents, the symbols they are monitoring.

*Note: `inputs_from` and `outputs_to` will define the edges in the flow visualization.*

## 5. Visualization on Dashboard (`AgentFlowGrid`)

The System Control Dashboard will feature an `AgentFlowGrid` component that visualizes this architecture:

*   **Nodes:** Each agent instance will be represented as a card (`AgentCard`).
    *   The card's appearance (e.g., color, icon) may vary based on `agent_role`.
    *   Key information like `name`, `status`, `type`, and relevant `metrics` will be displayed.
*   **Edges:** Lines will connect agents based on the `inputs_from` and `outputs_to` relationships, illustrating the flow of signals and directives.
    *   Edge styling (e.g., animated, color-coded) can indicate the status or nature of the data flow.
*   **Layout:** Nodes will be arranged logically, typically with Specialized Agents feeding into Decision Agent(s), which in turn feed into the Execution Layer.

This visualization will provide users with a clear and intuitive understanding of how the AI Trading Agent system operates and makes decisions.

## 6. Implementation Status (as of 2025-05-12)

This section tracks the progress of implementing the described architecture.

### 6.1. Core Agent Framework:
*   ✅ **Base Agent Class (`BaseAgent`):** Defined in `ai_trading_agent/agent/agent_definitions.py`, including common attributes like `agent_id`, `name`, `agent_role`, `status`, `inputs_from`, `outputs_to`, `config_details`, `metrics`, `last_updated`, `symbols`.
*   ✅ **Agent Roles Enum (`AgentRole`):** Defined to categorize agents (e.g., `specialized_sentiment`, `decision_aggregator`).
*   ✅ **Agent Status Enum (`AgentStatus`):** Defined for agent lifecycle states (e.g., `running`, `stopped`, `error`).
*   ✅ **Example Specialized Agents:**
    *   ✅ `SentimentAnalysisAgent` (basic structure)
    *   ✅ `TechnicalAnalysisAgent` (basic structure)
*   ✅ **Example Decision Agent (`DecisionAgent`):** Basic structure for signal aggregation and decision making.
*   ✅ **Example Execution Layer Agent (`ExecutionLayerAgent`):** Basic structure for handling trading directives.
*   ✅ **Trading Orchestrator (`TradingOrchestrator`):**
    *   ✅ Agent registration.
    *   ✅ Basic agent lifecycle management (start/stop all).
    *   [~] Initial dependency-based execution order determination. (Upgraded to Kahn's algorithm for topological sort)
    *   ✅ Basic data/signal routing between connected agents.
    *   ✅ Cycle-based processing.
*   ✅ **Python Package Structure:** `ai_trading_agent.agent` module created with `__init__.py`.
*   ✅ **Initial API Integration:**
    *   ✅ Orchestrator and example agents initialized in `api_server.py`.
    *   ✅ `/api/agents/status` endpoint to view agent information.
    *   ✅ `/api/agents/run_cycle` endpoint to manually trigger orchestrator cycle.

### 6.2. Specialized Agent Logic (Detailed Implementation):
*   `SentimentAnalysisAgent`:
    *   [~] Data fetching from news APIs/social media. (Basic simulation structure added)
    *   [~] Advanced sentiment scoring and trend analysis. (Basic simulation structure added)
*   `TechnicalAnalysisAgent`:
    *   [~] Real-time market data integration. (Basic simulation structure added)
    *   [~] Calculation of various technical indicators. (Basic simulation structure added)
    *   [ ] Chart pattern identification.
*   `NewsEventAgent`:
    *   [~] Design and implementation. (Basic class structure and simulated logic added)
*   `FundamentalAnalysisAgent`:
    *   [~] Design and implementation. (Basic class structure and simulated logic added)

### 6.3. Decision Agent Logic (Advanced):
*   [~] Configurable weighting/scoring for signals. (Basic structure for weights and thresholds added to `DecisionAgent`)
*   [~] Integration of risk management rules. (Basic quantity/value limits added to `DecisionAgent`)
*   [ ] Portfolio allocation and diversification logic.

### 6.4. Execution Layer (Broker Integration):
*   [~] Full integration with live/paper trading brokerage APIs. (Basic simulated structure added)
*   [~] Robust order management (placement, monitoring, cancellation). (Basic simulated structure added)
*   [~] Position management. (Basic simulated structure added)
*   [~] Detailed execution feedback loop. (Basic simulated structure added)

### 6.5. Dashboard Visualization (`AgentFlowGrid`):
*   [~] Frontend component development for `AgentCard`. (Exists and adapted for backend data structure)
*   [ ] Dynamic rendering of agent nodes and edges based on orchestrator data.
*   [ ] Real-time status updates and metric display.

### 6.6. Supporting Systems:
*   [ ] Advanced configuration management for agents.
*   [ ] Persistent storage for agent states/metrics (if needed beyond logging).
*   [ ] Comprehensive logging and monitoring for agent activities.
*   [ ] Robust error handling and recovery mechanisms within agents and orchestrator.
*   [~] Unit and integration tests for all components. (Initial tests for `BaseAgent`, `SentimentAnalysisAgent`, `TechnicalAnalysisAgent`, `NewsEventAgent`, `FundamentalAnalysisAgent`, `DecisionAgent` created)