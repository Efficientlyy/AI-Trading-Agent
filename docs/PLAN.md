# AI Trading Agent Rebuild Plan

## Overview
This document outlines the phased approach for rebuilding the AI Trading Agent with a focus on clean architecture, modern dashboard interface, and enhanced trading strategies.

## Project Status Overview
- ✅ **Phase 0: Clean Slate** - COMPLETED
- ✅ **Phase 1: Foundational Setup** - COMPLETED
- ✅ **Phase 2: Core Components** - COMPLETED
- 🔄 **Phase 3: Sentiment Analysis System** - IN PROGRESS
  - ✅ Phase 3.1: Sentiment Pipeline Integration Tests - COMPLETED
  - ✅ Phase 3.2: Sentiment Analysis Integration - COMPLETED
  - ✅ Phase 3.3: Agent Architecture Refactoring - COMPLETED
  - 🔄 Phase 3.4: Real Data Collectors Implementation - IN PROGRESS
  - 🔄 Phase 3.5: Advanced NLP Processing Pipeline - IN PROGRESS
  - 🔄 Phase 3.6: Advanced Trading Strategy Features - IN PROGRESS
- ⏳ **Phase 4: Genetic Algorithm Optimizer** - PENDING
- 🔄 **Phase 5: Multi-Asset Backtesting Framework** - PARTIALLY IMPLEMENTED
- ⏳ **Phase 6: Modern Dashboard Interface** - PENDING
- ⏳ **Phase 7: Integration and Deployment** - PENDING
- ⏳ **Phase 8: Continuous Improvement** - PENDING

---

## Detailed Phase Descriptions

### ✅ Phase 0: Clean Slate (COMPLETED)
- ✅ Create a new branch `rebuild-v2` for the rebuild process
- ✅ Remove all old source code, documentation, and configuration files
- ✅ Commit the clean slate to the repository

### ✅ Phase 1: Foundational Setup (COMPLETED)
- ✅ Create the basic directory structure:
  - `src/` for source code
  - `tests/` for test files
  - `config/` for configuration files
  - `docs/` for documentation
- ✅ Set up core dependencies in `requirements.txt` for Python 3.11
- ✅ Implement basic configuration management
- ✅ Create logging infrastructure
- ✅ Set up testing framework
- ✅ Create initial documentation
- ✅ Set up testing framework structure (`tests/unit`, `tests/integration`)
- ✅ Create initial documentation (`README.md`)
- ✅ Add `pytest.ini` for test configuration.
- ✅ Add placeholder documentation files (`architecture.md`, `usage_guide.md`, `contributing.md`) in `docs/`.

### ✅ Phase 2: Core Components (COMPLETED)
**Goal:** Implement the essential building blocks for data handling, processing, and trading execution.

**Tasks:**

1. ✅ **Implement Data Acquisition Module:**
   - ✅ Define `BaseDataProvider` interface (`src/data_acquisition/base_provider.py`).
   - ✅ Implement `MockDataProvider` for testing (`src/data_acquisition/mock_provider.py`).
   - ✅ Implement real data provider (e.g., using `ccxt` or `yfinance`) (`src/data_acquisition/ccxt_provider.py`).
   - ✅ Create `DataService` to manage providers based on configuration (`src/data_acquisition/data_service.py`).
   - ✅ Add unit tests for data providers.
   - ✅ Fix data provider tests and configuration mocking.

2. ✅ **Develop Data Processing Utilities:**
   - ✅ Implement functions for calculating technical indicators (e.g., SMA, EMA, RSI, MACD).
   - ✅ Create feature engineering pipeline.
   - ✅ Add unit tests for processing utilities.
   - ✅ Fix RSI calculation bug.

3. ✅ **Implement Trading Engine Core:**
   - ✅ Define core data models (`Order`, `Trade`, `Position`, `Portfolio`) using Pydantic (`src/trading_engine/models.py`).
   - ✅ Implement robust validation within models (Pydantic v2 compatible).
   - ✅ Create `OrderManager` for handling order lifecycle (`src/trading_engine/order_manager.py`).
   - ✅ Implement `PortfolioManager` for tracking positions and P&L (`src/trading_engine/portfolio_manager.py`).
   - ✅ Develop `ExecutionHandler` for simulating order execution (`src/trading_engine/execution_handler.py`).
   - ✅ Add unit tests for OrderManager (`tests/unit/trading_engine/test_order_manager.py`).
   - ⏳ Add unit tests for other managers and handlers.

4. ✅ **Develop Backtesting Framework:**
   - ✅ Implement basic backtesting loop (`src/backtesting/backtester.py`).
   - ✅ Define performance metrics calculation (`src/backtesting/performance_metrics.py`).

5. ✅ **Define Base Strategy Interface:**
   - ✅ Create `BaseStrategy` abstract class (`src/strategies/base_strategy.py`).
   - ✅ Implement simple example strategies (e.g., `MovingAverageCrossover`) (`src/strategies/ma_crossover.py`).
   - ✅ Add unit tests for strategies.

#### ✅ Additional Phase 2 Tasks (COMPLETED)

- ✅ **Finalize Trading Engine Integration Tests**
  - ✅ Refine timestamp-sensitive portfolio update tests
  - ✅ Cover stop and stop-limit order handling
  - ✅ Improve test coverage for order lifecycle and edge cases

- ✅ **Enhance Logging and Error Handling**
  - ✅ Add structured logging across modules
  - ✅ Improve error messages and exception handling

- ✅ **Improve Developer Documentation**
  - ✅ Update architecture diagrams
  - ✅ Document new APIs and integration points
  - ✅ Provide clear contribution guidelines

#### ✅ Rust Integration for Performance-Critical Components (COMPLETED)

**Phases:**

1. ✅ **Setup**
   - ✅ Install Rust toolchain and maturin
   - ✅ Create `rust_extensions/` with a Rust library crate
   - ✅ Configure `Cargo.toml` for PyO3 bindings

2. ✅ **Port Technical Indicators & Feature Engineering**
   - ✅ Use `ta-rs` and `ndarray` crates
   - ✅ Expose SMA, EMA, MACD as PyO3 functions
   - ✅ Expose RSI as PyO3 function
   - ✅ Implement lag features as PyO3 functions
   - ✅ Support NumPy array inputs/outputs
   - ✅ Replace Python implementations with Rust-backed calls
   - ✅ Validate with unit tests and benchmarks

3. ✅ **Accelerate Backtesting Core Loop**
   - ✅ Implement core simulation loop in Rust
   - ✅ Accept preprocessed features, initial portfolio, signals
   - ✅ Return trade logs, portfolio history, metrics
   - ✅ Expose via PyO3
   - ✅ Replace Python loop, validate correctness and speedup

4. ✅ **CI/CD Integration**
   - ✅ Automate Rust build with maturin
   - ✅ Run Rust and Python tests in CI
   - ✅ Ensure cross-platform builds

5. ✅ **Documentation**
   - ✅ Document Rust build, usage, and developer workflow
   - ✅ Provide benchmarks and integration examples

### 🔄 Phase 3: Sentiment Analysis System (IN PROGRESS)
- 🔄 Implement sentiment data collection from various sources
  - 🔄 Social media (Twitter, Reddit)
  - 🔄 News articles
  - 🔄 Market sentiment indicators (Fear & Greed Index)
- 🔄 Develop NLP processing pipeline
  - 🔄 Text preprocessing
  - 🔄 Sentiment scoring
  - 🔄 Entity recognition
- 🔄 Create sentiment-based trading strategy
  - 🔄 Signal generation based on sentiment thresholds
  - 🔄 Position sizing using volatility-based and Kelly criterion methods
  - 🔄 Stop-loss and take-profit management

#### 🔄 Detailed Sentiment Analysis Development Plan

1. ✅ **Design Modular Interfaces**
   - ✅ Define a `BaseSentimentProvider` abstract class with methods:
     - ✅ `fetch_sentiment_data()`
     - ✅ `stream_sentiment_data()`
   - ✅ Enables easy swapping of real, mock, or future providers.

2. ✅ **Implement MockSentimentProvider**
   - ✅ Generates synthetic sentiment data for initial integration and testing.
   - ✅ Returns sentiment scores, source metadata, and timestamps.

3. 🔄 **Plan Real Data Collectors**
   - 🔄 Design stubs for:
     - 🔄 Twitter API collector
     - 🔄 Reddit API collector
     - 🔄 News API collector
     - 🔄 Fear & Greed Index fetcher
   - 🔄 Implement incrementally, starting with public/free APIs.

4. 🔄 **Develop NLP Processing Pipeline**
   - 🔄 Create `TextPreprocessor` for cleaning and normalizing text
   - 🔄 Implement `SentimentAnalyzer` with multiple models:
     - 🔄 Rule-based (VADER)
     - 🔄 ML-based (DistilBERT or similar)
   - 🔄 Add `EntityRecognizer` for identifying assets/tickers
   - 🔄 Create unit tests for each component

5. 🔄 **Build Signal Generation**
   - 🔄 Create `SentimentSignalGenerator` class
   - 🔄 Implement time-based aggregation of sentiment scores
   - 🔄 Add configurable thresholds for signal generation
   - 🔄 Create visualization tools for sentiment trends

#### ✅ Phase 3.1: Sentiment Pipeline Integration Tests (COMPLETED)

**Scope:**
- ✅ Connect sentiment data collection, NLP processing, feature generation, signal generation, order creation, and portfolio updates into an end-to-end flow.
- ✅ Develop integration tests covering this entire pipeline.
- ✅ Validate that sentiment-driven signals produce expected trading behavior.

**Tasks:**

1. ✅ **Create Integration Test File (`tests/integration/test_sentiment_pipeline_integration.py`):**
   - ✅ Set up test fixtures for mock data providers
   - ✅ Create test scenarios covering different sentiment patterns

2. ✅ **Test Scenario 1: Positive Sentiment Entry:**
   - ✅ **Goal:** Verify buy signal generation and execution.
   - ✅ **Setup:** Mock market data, mock sentiment (consistently above buy threshold), initial cash.
   - ✅ **Verification:** Assert BUY order generated, final portfolio has position.

3. ✅ **Test Scenario 2: Negative Sentiment Exit:**
   - ✅ **Goal:** Verify sell signal generation and execution.
   - ✅ **Setup:** Mock market data, mock sentiment (consistently below sell threshold), initial *long* position.
   - ✅ **Verification:** Assert SELL order generated, final portfolio quantity is zero.

4. ✅ **Test Scenario 3: Sentiment Reversal Handling (Positive -> Negative):**
   - ✅ **Goal:** Ensure correct position flip.
   - ✅ **Setup:** Mock market data, mock sentiment reversing from positive to negative, initial cash.
   - ✅ **Verification:** Assert BUY followed by SELL orders, final portfolio matches expectations.

5. ✅ **Test Scenario 4: Sentiment Threshold Sensitivity:**
   - ✅ **Goal:** Verify thresholds work as expected.
   - ✅ **Setup:** Mock market data, mock sentiment hovering near thresholds, initial cash.
   - ✅ **Verification:** Assert orders only generated when thresholds crossed.

6. ✅ **Test Scenario 5: Multi-Source Sentiment Aggregation:**
   - ✅ **Goal:** Ensure proper weighting of different sentiment sources.
   - ✅ **Setup:** Mock market data, multiple sentiment sources with different weights, initial cash.
   - ✅ **Verification:** Assert final signal matches expected weighted average.

7. ✅ **Test Scenario 6: Multi-Asset Sentiment Strategy:**
   - ✅ **Goal:** Verify correct handling of multiple assets with different sentiment.
   - ✅ **Setup:** Mock market data for assets A, B, C; different sentiment patterns for each.
   - ✅ **Verification:** Use `MultiAssetBacktester` with `equal_weight_allocation`, assert final positions match sentiment signals (long A, flat/short B, flat C).

8. 🔄 **(Optional) Test Scenario 7: Sentiment-Weighted Allocation:**
   - 🔄 **Goal:** Verify sentiment-based allocation weights shift correctly.
   - 🔄 **Setup:** Similar to Scenario 6, designed to cause allocation changes.
   - 🔄 **Verification:** Assert allocation weights correlate with sentiment strength.

**Implementation Strategy:**
- ✅ Create fixtures in `conftest.py` for reusable components
- ✅ Implement each test scenario in separate test functions
- ✅ Use parameterization for threshold sensitivity tests
- ✅ **Next:** Implement integration tests and iterate

#### ✅ Phase 3.2: Sentiment Analysis Integration (COMPLETED)

**Goal:** Integrate a sentiment analysis pipeline to generate trading signals for the backtesting engine.

**Detailed Steps:**

1. ✅ **Create Sentiment Analysis Module Structure:**
   - ✅ Create directory: `ai_trading_agent/sentiment_analysis/`
   - ✅ Add initial files: `__init__.py`, `analyzer.py`, `utils.py` (optional).

2. ✅ **Implement SentimentAnalyzer Class:**
   - ✅ Create `SentimentAnalyzer` class in `analyzer.py`.
   - ✅ Implement methods:
     - ✅ `analyze_text(text: str) -> float`: Returns sentiment score for a text.
     - ✅ `analyze_batch(texts: List[str]) -> List[float]`: Batch processing.
   - ✅ Support multiple backends (VADER, Transformers).
   - ✅ Add configuration options for model selection.

3. ✅ **Create SignalGenerator Class:**
   - ✅ Implement `SignalGenerator` class.
   - ✅ Add methods:
     - ✅ `generate_signals(sentiment_data: pd.DataFrame) -> pd.DataFrame`
   - ✅ Support configurable thresholds and time windows.
   - ✅ Return DataFrame with signal columns (-1, 0, 1).

4. ✅ **Develop SentimentStrategy:**
   - ✅ Create `SentimentStrategy` class inheriting from `BaseStrategy`.
   - ✅ Implement required methods:
     - ✅ `generate_signals(market_data: pd.DataFrame) -> pd.DataFrame`
     - ✅ `calculate_positions(signals: pd.DataFrame, portfolio: Portfolio) -> Dict[str, float]`
   - ✅ Use `SentimentAnalyzer` and `SignalGenerator` internally.

5. ✅ **Create Integration Test:**
   - ✅ Implement `test_sentiment_strategy_integration.py`.
   - ✅ Test full pipeline:
     - ✅ Load sample text data via connector.
     - ✅ Analyze text with `SentimentAnalyzer`.
     - ✅ Generate signals with `SignalGenerator`.
   - ✅ Ensure signals align with OHLCV timestamps.
   - ✅ Use generated signals to create orders.

#### ✅ Phase 3.3: Agent Architecture Refactoring (COMPLETED)

**Goal:** Refactor the trading agent into a modular architecture with clear separation of concerns.

**Components:**

1. ✅ **Data Manager:**
   - ✅ **Responsibility:** Acquires and preprocesses market data from various sources.
   - ✅ **Integration:** Provides clean, normalized data to strategies.

2. ✅ **Strategy Manager:**
   - ✅ **Responsibility:** Hosts strategies and coordinates signal generation.
   - ✅ **Integration:** Hosts strategies like the `SentimentStrategy` (using `SentimentAnalyzer` and `SentimentSignalGenerator`). Will manage combining signals if multiple strategies are active.

3. ✅ **Portfolio Manager:**
   - ✅ **Responsibility:** Takes trading *signals*. Considers current portfolio state, risk constraints (from Risk Manager), and position sizing rules. Translates signals into concrete *order* requests.
   - ✅ **Integration:** Interfaces with `Strategy` for signals and `RiskManager` for constraints.

4. ✅ **Risk Manager:**
   - ✅ **Responsibility:** Enforces risk limits and constraints. Provides position sizing recommendations based on volatility, correlation, and other risk metrics.
   - ✅ **Integration:** Used by `PortfolioManager` to determine safe position sizes.

5. ✅ **Execution Handler:**
   - ✅ **Responsibility:** Takes *order* requests from Portfolio Manager. Handles execution details (market/limit orders, etc.). Returns *fill* information.
   - ✅ **Integration:** Receives orders from `PortfolioManager`, interfaces with exchange or broker.

6. ✅ **Orchestrator:**
   - ✅ **Responsibility:** Coordinates the flow between all components. Manages the trading lifecycle.
   - ✅ **Integration:** Central component that ties everything together.

**Implementation Steps:**

1. ✅ Define interfaces for each component (`DataManager`, `Strategy`, `PortfolioManager`, `RiskManager`, `ExecutionHandler`, `Orchestrator`).
2. ✅ Implement each component with proper separation of concerns.
3. ✅ Implement the `Orchestrator` to manage the flow.
4. ✅ Develop a configuration system for the new architecture.
5. ✅ Update tests to reflect the new structure.

### 🔄 Phase 3.4: Real Data Collectors Implementation (IN PROGRESS)
- 🔄 Implement real data collectors for various sources:
  - ✅ Twitter API collector
    - ✅ Set up Twitter API authentication
    - ✅ Implement tweet search and streaming
    - ✅ Add filtering by keywords, hashtags, and users
    - ✅ Implement rate limiting and error handling
    - ✅ Create comprehensive unit tests
  - 🔄 Reddit API collector
    - 🔄 Set up Reddit API authentication
    - 🔄 Implement subreddit and post search
    - 🔄 Add comment extraction and analysis
    - 🔄 Implement rate limiting and error handling
  - ✅ News API collector
    - ✅ Integrate with financial news APIs
    - ✅ Implement article search and filtering
    - ✅ Add content extraction and cleaning
    - ✅ Implement caching and rate limiting
    - ✅ Create comprehensive unit tests
  - ✅ Fear & Greed Index fetcher
    - ✅ Implement data scraping for Fear & Greed Index
    - ✅ Add historical data retrieval
    - ✅ Implement normalization and integration with other data
    - ✅ Create comprehensive unit tests

### 🔄 Phase 3.5: Advanced NLP Processing Pipeline (IN PROGRESS)
- 🔄 Enhance text preprocessing
  - 🔄 Implement advanced tokenization
  - 🔄 Add named entity recognition for financial terms
  - 🔄 Implement text normalization techniques
  - 🔄 Add support for multiple languages
- 🔄 Improve sentiment analysis
  - 🔄 Integrate domain-specific sentiment models
  - 🔄 Implement fine-tuning on financial text
  - 🔄 Add context-aware sentiment analysis
  - 🔄 Implement ensemble methods for higher accuracy
- 🔄 Add entity recognition
  - 🔄 Implement company and ticker symbol recognition
  - 🔄 Add financial metric and event detection
  - 🔄 Implement relationship extraction between entities
  - 🔄 Add confidence scoring for entity matching

### 🔄 Phase 3.6: Advanced Trading Strategy Features (IN PROGRESS)
- 🔄 Enhance signal generation
  - 🔄 Implement adaptive sentiment thresholds
  - 🔄 Add time-series analysis of sentiment trends
  - 🔄 Implement sentiment momentum indicators
  - 🔄 Add cross-asset sentiment correlation analysis
- 🔄 Implement advanced position sizing
  - 🔄 Add Kelly criterion position sizing
  - 🔄 Implement volatility-adjusted position sizing
  - 🔄 Add correlation-based portfolio allocation
  - 🔄 Implement risk parity approaches
- 🔄 Add sophisticated risk management
  - 🔄 Implement dynamic stop-loss placement
  - 🔄 Add take-profit optimization
  - 🔄 Implement trailing stop mechanisms
  - 🔄 Add drawdown-based position reduction

### ⏳ Phase 4: Genetic Algorithm Optimizer (PENDING)
- ⏳ Implement parameter optimization framework
  - ⏳ Fitness function definition
  - ⏳ Population management
  - ⏳ Crossover and mutation operations
- ⏳ Develop strategy comparison capabilities
  - ⏳ Performance metrics calculation
  - ⏳ Strategy evaluation
- ⏳ Create realistic market condition simulation
  - ⏳ Transaction costs
  - ⏳ Market biases
  - ⏳ Slippage modeling

### 🔄 Phase 5: Multi-Asset Backtesting Framework (PARTIALLY IMPLEMENTED)
- 🔄 Implement portfolio-level backtesting
  - 🔄 Asset allocation
  - 🔄 Correlation analysis
  - 🔄 Risk management across the entire portfolio
- 🔄 Develop performance metrics for portfolio evaluation
- 🔄 Create visualization tools for portfolio performance

### ⏳ Phase 6: Modern Dashboard Interface (PENDING)
- ⏳ Design and implement a modular dashboard
  - ⏳ Trading overview
  - ⏳ Strategy performance
  - ⏳ Sentiment analysis visualization
  - ⏳ Portfolio management
- ⏳ **Backend API tasks:**
  - ⏳ Create REST API endpoints to serve portfolio data, strategy performance, sentiment data
  - ⏳ Implement endpoints for order management and parameter adjustments
  - ⏳ Support real-time data streaming (e.g., websockets)
  - ⏳ Implement user authentication and authorization
  - ⏳ Provide API for backtesting controls and results retrieval
- ⏳ Create interactive frontend components
  - ⏳ Strategy parameter adjustment
  - ⏳ Backtesting controls
  - ⏳ Real-time monitoring
- ⏳ Implement authentication and security features
- ⏳ **Backend API Enhancements:**
  - ⏳ Implement environment variable loading from `.env`
  - ⏳ Complete authentication flow with JWT refresh, password reset
  - ⏳ Add asset information endpoints
  - ⏳ Add historical data endpoints with flexible timeframes
  - ⏳ Add detailed performance metrics endpoints
  - ⏳ Add strategy management endpoints (CRUD)
  - ⏳ Implement background job queue for backtests and long tasks
  - ⏳ Enhance websocket with subscription model
  - ⏳ Integrate database for users, strategies, backtest results
  - ⏳ Implement comprehensive error handling
  - ⏳ Add unit and integration tests for API endpoints
- ⏳ **Frontend Development Stages:**
  - ⏳ **Stage 1: Foundation**
    - ⏳ Set up React/TypeScript project structure and architecture
    - ⏳ Implement authentication components (login, registration)
    - ⏳ Create responsive layout with navigation sidebar
    - ⏳ Set up API client services and token management
    - ⏳ Implement WebSocket connection with subscription management
  - ⏳ **Stage 2: Dashboard Core**
    - ⏳ Develop portfolio summary widgets
    - ⏳ Create performant chart components (equity curve, allocation)
    - ⏳ Build trading interface with order entry form
    - ⏳ Implement real-time data updates via WebSockets
    - ⏳ Create unified notification system
  - ⏳ **Stage 3: Trading Tools**
    - ⏳ Build technical analysis chart with indicators
    - ⏳ Implement order management interface
    - ⏳ Create position management visualization
    - ⏳ Develop risk calculator components
    - ⏳ Add sentiment signal indicators
  - ⏳ **Stage 4: Advanced Features**
    - ⏳ Implement strategy builder and configuration
    - ⏳ Create backtest configuration and results visualization
    - ⏳ Build comparative analysis tools for strategies
    - ⏳ Develop performance metrics dashboard
    - ⏳ Add historical data visualization tools
  - ⏳ **Stage 5: Polish and Optimization**
    - ⏳ Optimize rendering performance
    - ⏳ Implement theme customization (dark/light modes)
    - ⏳ Add responsive adaptations for all device sizes
    - ⏳ Create animation system for transitions
    - ⏳ Implement comprehensive error handling
    - ⏳ Complete end-to-end testing

### ⏳ Phase 7: Integration and Deployment (PENDING)
- ⏳ Connect to real trading APIs
- ⏳ Implement paper trading mode
- ⏳ Set up continuous integration and testing
- ⏳ Create deployment documentation
- ⏳ Implement monitoring and alerting

### ⏳ Phase 8: Continuous Improvement (PENDING)
- ⏳ Performance optimization
- ⏳ Additional trading strategies
- ⏳ Enhanced visualization features
- ⏳ User feedback integration
