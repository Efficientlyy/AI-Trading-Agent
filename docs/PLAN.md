# AI Trading Agent Rebuild Plan

## Overview
This document outlines the phased approach for rebuilding the AI Trading Agent with a focus on clean architecture, modern dashboard interface, and enhanced trading strategies.

## Phase 0: Clean Slate (Completed)
- Create a new branch `rebuild-v2` for the rebuild process
- Remove all old source code, documentation, and configuration files
- Commit the clean slate to the repository

## Phase 1: Foundational Setup
- Create the basic directory structure:
  - `src/` for source code
  - `tests/` for test files
  - `config/` for configuration files
  - `docs/` for documentation
- Set up core dependencies in `requirements.txt` for Python 3.11
- Implement basic configuration management
- Create logging infrastructure
- Set up testing framework
- Create initial documentation
*   [x] Set up testing framework structure (`tests/unit`, `tests/integration`)
*   [x] Create initial documentation (`README.md`)
*   [x] Add `pytest.ini` for test configuration.
*   [x] Add placeholder documentation files (`architecture.md`, `usage_guide.md`, `contributing.md`) in `docs/`.

**Status: Completed**

## Phase 2: Core Components
**Goal:** Implement the essential building blocks for data handling, processing, and trading execution.

**Tasks:**

1.  **Implement Data Acquisition Module:**
    *   [x] Define `BaseDataProvider` interface (`src/data_acquisition/base_provider.py`).
    *   [x] Implement `MockDataProvider` for testing (`src/data_acquisition/mock_provider.py`).
    *   [x] Implement real data provider (e.g., using `ccxt` or `yfinance`) (`src/data_acquisition/ccxt_provider.py`).
    *   [x] Create `DataService` to manage providers based on configuration (`src/data_acquisition/data_service.py`).
    *   [x] Add unit tests for data providers.
    *   [x] Fix data provider tests and configuration mocking.
2.  **Develop Data Processing Utilities:**
    *   [x] Implement functions for calculating technical indicators (e.g., SMA, EMA, RSI, MACD).
    *   [x] Create feature engineering pipeline.
    *   [x] Add unit tests for processing utilities.
    *   [x] Fix RSI calculation bug.

---

### **Additional Phase 2 Tasks (Prerequisites for Phase 3)**

- **Finalize Trading Engine Integration Tests** [x]
  - Refine timestamp-sensitive portfolio update tests
  - Cover stop and stop-limit order handling
  - Improve test coverage for order lifecycle and edge cases

- **Enhance Logging and Error Handling** [x]
  - Add structured logging across modules
  - Improve error messages and exception handling

- **Improve Developer Documentation**
  - Update architecture diagrams
  - Document new APIs and integration points
  - Provide clear contribution guidelines

These improvements will solidify the core system before integrating sentiment analysis in Phase 3.

---

### **Rust Integration for Performance-Critical Components**

To accelerate data processing and backtesting, we will incrementally integrate Rust using PyO3/maturin. This plan ensures maintainability and leverages mature Rust tooling.

**Phases:**

**1. Setup**

- Install Rust toolchain and maturin
- Create `rust_extensions/` with a Rust library crate
- Configure `Cargo.toml` for PyO3 bindings

**2. Port Technical Indicators & Feature Engineering**

- [x] Use `ta-rs` and `ndarray` crates
- [x] Expose SMA, EMA, MACD as PyO3 functions
- [x] Expose RSI as PyO3 function
- [x] Implement lag features as PyO3 functions
- [x] Support NumPy array inputs/outputs
- [x] Replace Python implementations with Rust-backed calls
- [x] Validate with unit tests and benchmarks

**3. Accelerate Backtesting Core Loop**

- [x] Implement core simulation loop in Rust
- [x] Accept preprocessed features, initial portfolio, signals
- [x] Return trade logs, portfolio history, metrics
- [x] Expose via PyO3
- [x] Replace Python loop, validate correctness and speedup

**4. CI/CD Integration**

- Automate Rust build with maturin
- Run Rust and Python tests in CI
- Ensure cross-platform builds

**5. Documentation**

- Document Rust build, usage, and developer workflow
- Provide benchmarks and integration examples

---

This integration will **significantly improve performance** of data processing and backtesting, while keeping the rest of the system in Python for flexibility.

3.  **Implement Trading Engine Core:**
    *   [x] Define core data models (`Order`, `Trade`, `Position`, `Portfolio`) using Pydantic (`src/trading_engine/models.py`).
    *   [x] Implement robust validation within models (Pydantic v2 compatible).
    *   [x] Add essential methods to models (e.g., `Position.update_position`, `Portfolio.update_from_trade`, `Order.get_average_fill_price`).
    *   [x] Add comprehensive unit tests for all models (`tests/unit/trading_engine/test_models.py`).
    *   [x] Implement `OrderManager` for order lifecycle management (`src/trading_engine/order_manager.py`).
    *   [x] Fix PnL calculation with standalone function to resolve TypeError issues.
    *   [x] Add `average_fill_price` property to Order class.
    *   [x] Fix parameter name mismatches in Order.add_fill method calls.
    *   [x] Implement proper handling of finalized orders.
    *   [x] Implement `ExecutionHandler` to simulate trade execution (`src/trading_engine/execution_handler.py` - previously `execution_simulator.py`).
    *   [x] Implement `PortfolioManager` logic (partially covered by `Portfolio` model methods, may need a dedicated manager later).
    *   [x] Add unit tests for OrderManager (`tests/unit/trading_engine/test_order_manager.py`).
    *   [ ] Add unit tests for other managers and handlers.
4.  **Develop Backtesting Framework:**
    *   [x] Implement basic backtesting loop (`src/backtesting/backtester.py`).
    *   [x] Define performance metrics calculation (`src/backtesting/performance_metrics.py`).
5.  **Define Base Strategy Interface:**
    *   [x] Create `BaseStrategy` class with methods for initialization and generating signals (`generate_signals`).
    *   [x] Add basic example strategy implementation.

**Status: In Progress**

## Phase 3: Sentiment Analysis System
- Implement sentiment data collection from various sources
  - Social media (Twitter, Reddit)
  - News articles
  - Market sentiment indicators (Fear & Greed Index)
- Develop NLP processing pipeline
  - Text preprocessing
  - Sentiment scoring
  - Entity recognition
- Create sentiment-based trading strategy
  - Signal generation based on sentiment thresholds
  - Position sizing using volatility-based and Kelly criterion methods
  - Stop-loss and take-profit management
---

### **Detailed Sentiment Analysis Development Plan**

1. **Design Modular Interfaces**
   - Define a `BaseSentimentProvider` abstract class with methods:
     - `fetch_sentiment_data()`
     - `stream_sentiment_data()`
   - Enables easy swapping of real, mock, or future providers.

2. **Implement MockSentimentProvider**
   - Generates synthetic sentiment data for initial integration and testing.
   - Returns sentiment scores, source metadata, and timestamps.

3. **Plan Real Data Collectors**
   - Design stubs for:
     - Twitter API collector
     - Reddit API collector
     - News API collector
     - Fear & Greed Index fetcher
   - Implement incrementally, starting with public/free APIs.

4. **Develop NLP Pipeline**
   - Text cleaning and normalization
   - Sentiment scoring (VADER, TextBlob, or transformer models)
   - Entity recognition (optional)

5. **Integrate with Trading Pipeline**
   - Feed sentiment scores into market regime detection and strategy signal generation.
   - Use mock data initially, then real data.

6. **Testing**
   - Create unit tests for each provider and NLP component.
   - Develop integration tests simulating end-to-end sentiment-driven trading.

7. **Documentation**
   - Document interfaces, data formats, and integration points.
   - Update architecture diagrams accordingly.

This modular, test-driven approach ensures incremental, reliable progress on the sentiment analysis system.


### Phase 3.1: Sentiment Analysis Integration & Testing

**Goal:** Fully integrate the sentiment analysis system into the trading pipeline and develop comprehensive integration tests to ensure correctness.

**Scope:**

- Connect sentiment data collection, NLP processing, feature generation, signal generation, order creation, and portfolio updates into an end-to-end flow.
- Develop integration tests covering this entire pipeline.
- Validate that sentiment-driven signals produce expected trading behavior.

**Tasks:**

1. **Design Integration Test Plan**
   - Define test scenarios using mock sentiment and market data.
   - Specify expected signals, orders, and portfolio changes.

2. **Prepare Test Fixtures**
   - Create mock sentiment datasets with known patterns.
   - Generate or select static market price data.
   - Initialize portfolio states and configurations.

3. **Implement Integration Tests**
   - Test feature generation correctness.
   - Verify signal generation logic.
   - Confirm order creation and risk management.
   - Simulate order fills and portfolio updates.
   - Run full pipeline simulations over time windows.

4. **Run and Debug**
   - Execute tests, analyze failures.
   - Fix issues in sentiment service or trading engine as needed.
   - Iterate until all tests pass.

5. **Document Results**
   - Summarize test coverage and outcomes.
   - Update usage and developer docs accordingly.

**Expected Outcomes:**

- Confidence that sentiment signals are correctly integrated.
- A robust test suite for future development.
- Clear documentation of the sentiment-driven trading flow.

**Status:**

- **Planned** (to be started after this update)
- **Dependencies:** Core trading engine and sentiment modules implemented
- **Next:** Implement integration tests and iterate

## Phase 4: Genetic Algorithm Optimizer
- Implement parameter optimization framework
  - Fitness function definition
  - Population management
  - Crossover and mutation operations
- Develop strategy comparison capabilities
  - Performance metrics calculation
  - Strategy evaluation
- Create realistic market condition simulation
  - Transaction costs
  - Market biases
  - Slippage modeling

## Phase 5: Multi-Asset Backtesting Framework
- Implement portfolio-level backtesting
  - Asset allocation
  - Correlation analysis
  - Risk management across the entire portfolio
- Develop performance metrics for portfolio evaluation
- Create visualization tools for portfolio performance

## Phase 6: Modern Dashboard Interface
- Design and implement a modular dashboard
  - Trading overview
  - Strategy performance
  - Sentiment analysis visualization
  - Portfolio management
- **Backend API tasks:**
  - Create REST API endpoints to serve portfolio data, strategy performance, sentiment data
  - Implement endpoints for order management and parameter adjustments
  - Support real-time data streaming (e.g., websockets)
  - Implement user authentication and authorization
  - Provide API for backtesting controls and results retrieval
- Create interactive frontend components
  - Strategy parameter adjustment
  - Backtesting controls
  - Real-time monitoring
- Implement authentication and security features
- **Backend API Enhancements:**
  - Implement environment variable loading from `.env`
  - Complete authentication flow with JWT refresh, password reset
  - Add asset information endpoints
  - Add historical data endpoints with flexible timeframes
  - Add detailed performance metrics endpoints
  - Add strategy management endpoints (CRUD)
  - Implement background job queue for backtests and long tasks
  - Enhance websocket with subscription model
  - Integrate database for users, strategies, backtest results
  - Implement comprehensive error handling
  - Add unit and integration tests for API endpoints
- **Frontend Development Stages:**
  - **Stage 1: Foundation**
    - Set up React/TypeScript project structure and architecture
    - Implement authentication components (login, registration)
    - Create responsive layout with navigation sidebar
    - Set up API client services and token management
    - Implement WebSocket connection with subscription management
  - **Stage 2: Dashboard Core**
    - Develop portfolio summary widgets
    - Create performant chart components (equity curve, allocation)
    - Build trading interface with order entry form
    - Implement real-time data updates via WebSockets
    - Create unified notification system
  - **Stage 3: Trading Tools**
    - Build technical analysis chart with indicators
    - Implement order management interface
    - Create position management visualization
    - Develop risk calculator components
    - Add sentiment signal indicators
  - **Stage 4: Advanced Features**
    - Implement strategy builder and configuration
    - Create backtest configuration and results visualization
    - Build comparative analysis tools for strategies
    - Develop performance metrics dashboard
    - Add historical data visualization tools
  - **Stage 5: Polish and Optimization**
    - Optimize rendering performance
    - Implement theme customization (dark/light modes)
    - Add responsive adaptations for all device sizes
    - Create animation system for transitions
    - Implement comprehensive error handling
    - Complete end-to-end testing

## Phase 7: Integration and Deployment
- Connect to real trading APIs
- Implement paper trading mode
- Set up continuous integration and testing
- Create deployment documentation
- Implement monitoring and alerting

## Phase 8: Continuous Improvement
- Performance optimization
- Additional trading strategies
- Enhanced visualization features
- User feedback integration
