# AI Trading Agent Rebuild Plan

## Dashboard & Trade Page Refactor Plan

To align with the modular, multi-agent architecture and provide a professional user experience, the dashboard and trading functionality will be split into dedicated pages. This separation enhances clarity, performance, and future scalability.

### 1. Page Responsibilities

**Dashboard Page (Overview & Monitoring):**
- ✅ Portfolio Summary (value, allocation, performance)
- ✅ Asset Allocation Chart (clickable, drill-down)
- ✅ Technical Analysis Chart (overview mode, key indicators)
- ✅ Sentiment Summary (signal strengths, news/social/fundamental signals)
- ✅ Recent Trades (latest trades, quick status)
- ✅ Notifications (alerts, errors, info)
- ✅ Quick Links (Trading, Backtesting, Strategies, Settings)

**Trade Page (Action & Execution):**
- ✅ Asset Selector (context-aware, defaults to asset clicked in dashboard)
- ✅ Live Price Chart (focused, trading indicators)
- ✅ Order Entry Form (buy/sell, order type, quantity, price, etc.)
- ✅ Order Book & Recent Trades (for selected asset)
- ✅ Open Orders & Order History (manage/cancel/view trades)
- ✅ Position Details (current position, P&L, risk for selected asset)
- ✅ Trade Confirmation & Feedback

### 2. File/Component Structure
- `/src/pages/Dashboard.tsx` — high-level monitoring/overview
- `/src/pages/Trade.tsx` — all trading-specific components and logic
- `/src/components/dashboard/` — PortfolioSummary, AssetAllocationChart, TechnicalChart, SentimentSummary, RecentTrades, Notifications, etc.
- `/src/components/trading/` — OrderEntryForm, OrderBook, TradeHistory, PositionDetails, AssetSelector, etc.

### 3. Routing
- Add `/dashboard` and `/trade` (or `/trading`) routes in `App.tsx`
- Pass selected asset as route state or via context

### 4. State & Context
- Use React Context or global state for:
  - Selected asset/symbol
  - Notifications/messages

### 5. E2E Test Updates
- Update Cypress tests for new navigation and page structure
- Test dashboard and trade page flows independently

### 6. Documentation
- Update PLAN.md to reflect:
  - Separation of concerns
  - Component locations
  - Rationale for the new structure

### Multi-Agent System
- **Specialized Agents:** Each agent analyzes a unique data stream or trading strategy (e.g., technical indicators, sentiment, news, fundamentals) and produces trading signals or insights.
- **Decision Agent:** Aggregates all agent signals, applies risk management and portfolio constraints, and determines final trading actions (buy/sell/hold, sizing, etc.).
- **Execution Layer:** Handles order placement, monitoring, and feedback to agents for learning and adaptation.

### Why This Approach?
- **Modularity:** Agents can be developed, tested, and improved independently.
- **Adaptivity:** New agents can be added as new data sources or strategies emerge.
- **Transparency:** The system can explain which signals led to a decision.
- **Robustness:** Reduces reliance on any single strategy or data stream.

The platform includes a modern dashboard for monitoring portfolio, agent signals, trades, and analytics, as well as dedicated pages for trading, backtesting, and strategy management.

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
  - 🔄 Phase 3.7: Sentiment-Trading Engine Integration - IN PROGRESS
- ✅ **Phase 4: Genetic Algorithm Optimizer** - COMPLETED
- ✅ **Phase 5: Multi-Asset Backtesting Framework** - COMPLETED
- ✅ **Phase 6: Modern Dashboard Interface** - COMPLETED
- ⏳ **Phase 7: Integration and Deployment** - PENDING
- 🔄 **Phase 8: Continuous Improvement** - IN PROGRESS

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
   - ✅ Implement `PortfolioManager` (`src/trading_engine/portfolio_manager.py`).
   - ✅ **Refactor Trading Engine for Decimal Precision:**
     - ✅ Models (`Position`, `Portfolio`) updated to use `Decimal`.
     - ✅ `PortfolioManager` updated to use `Decimal`.
     - ✅ Fixed compatibility issues in `Portfolio.update_from_trade`.
   - ✅ Implement basic order execution logic.
   - ✅ Add unit tests for trading engine components.

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
  - ✅ Social media (Reddit)
  - ✅ Social media (Twitter)
  - ✅ News articles
  - ✅ Market sentiment indicators (Fear & Greed Index)
- ✅ Develop NLP processing pipeline
  - ✅ Text preprocessing
  - ✅ Sentiment scoring
  - ✅ Entity recognition
- ✅ Create sentiment-based trading strategy
  - ✅ Signal generation based on sentiment thresholds
  - ✅ Position sizing using volatility-based and Kelly criterion methods
  - ✅ Stop-loss and take-profit management

#### 🔄 Detailed Sentiment Analysis Development Plan

1. ✅ **Design Modular Interfaces**
   - ✅ Define a `BaseSentimentProvider` abstract class with methods:
     - ✅ `fetch_sentiment_data()`
     - ✅ `stream_sentiment_data()`
   - ✅ Enables easy swapping of real, mock, or future providers.

2. ✅ **Implement MockSentimentProvider**
   - ✅ Generates synthetic sentiment data for initial integration and testing.
   - ✅ Returns sentiment scores, source metadata, and timestamps.

3. ✅ **Plan Real Data Collectors**
   - ✅ Design stubs for:
     - ✅ Twitter API collector
     - ✅ Reddit API collector
     - ✅ News API collector
     - ✅ Fear & Greed Index fetcher
   - ✅ Implement incrementally, starting with public/free APIs.

4. 🔄 **Develop NLP Processing Pipeline**
   - ✅ Create `TextPreprocessor` for cleaning and normalizing text
   - ✅ Implement `SentimentAnalyzer` with multiple models:
     - ✅ Rule-based (VADER)
     - ✅ ML-based (DistilBERT or similar)
   - ✅ Add `EntityRecognizer` for identifying assets/tickers
   - ✅ Create unit tests for each component

5. ✅ **Build Signal Generation**
   - ✅ Create `SentimentSignalGenerator` class
   - ✅ Implement time-based aggregation of sentiment scores
   - ✅ Add configurable thresholds for signal generation
   - ✅ Create visualization tools for sentiment trends

#### ✅ Phase 3.1: Sentiment Pipeline Integration Tests (COMPLETED)

**Scope:**
- ✅ Connect sentiment data collection, NLP processing, feature generation, signal generation, order creation, and portfolio updates into an end-to-end flow.
- ✅ Develop integration tests covering this entire pipeline.
- ✅ Validate that sentiment-driven signals produce expected trading behavior.

**Tasks:**
- ✅ Create Integration Test File (`tests/integration/test_sentiment_pipeline_integration.py`):
  - ✅ Set up test fixtures for mock data providers
  - ✅ Create test scenarios covering different sentiment patterns

- ✅ Test Scenario 1: Positive Sentiment Entry:
  - ✅ Goal: Verify buy signal generation and execution.
  - ✅ Setup: Mock market data, mock sentiment (consistently above buy threshold), initial cash.
  - ✅ Verification: Assert BUY order generated, final portfolio has position.

- ✅ Test Scenario 2: Negative Sentiment Exit:
  - ✅ Goal: Verify sell signal generation and execution.
  - ✅ Setup: Mock market data, mock sentiment (consistently below sell threshold), initial *long* position.
  - ✅ Verification: Assert SELL order generated, final portfolio quantity is zero.

- ✅ Test Scenario 3: Sentiment Reversal Handling (Positive -> Negative):
  - ✅ Goal: Ensure correct position flip.
  - ✅ Setup: Mock market data, mock sentiment reversing from positive to negative, initial cash.
  - ✅ Verification: Assert BUY followed by SELL orders, final portfolio matches expectations.

- ✅ Test Scenario 4: Sentiment Threshold Sensitivity:
  - ✅ Goal: Verify thresholds work as expected.
  - ✅ Setup: Mock market data, mock sentiment hovering near thresholds, initial cash.
  - ✅ Verification: Assert orders only generated when thresholds crossed.

- ✅ Test Scenario 5: Multi-Source Sentiment Aggregation:
  - ✅ Goal: Ensure proper weighting of different sentiment sources.
  - ✅ Setup: Mock market data, multiple sentiment sources with different weights, initial cash.
  - ✅ Verification: Assert final signal matches expected weighted average.

- ✅ Test Scenario 6: Multi-Asset Sentiment Strategy:
  - ✅ Goal: Verify correct handling of multiple assets with different sentiment.
  - ✅ Setup: Mock market data for assets A, B, C; different sentiment patterns for each.
  - ✅ Verification: Use `MultiAssetBacktester` with `equal_weight_allocation`, assert final positions match sentiment signals (long A, flat/short B, flat C).

- 🔄 (Optional) Test Scenario 7: Sentiment-Weighted Allocation:
  - 🔄 Goal: Verify sentiment-based allocation weights shift correctly.
  - 🔄 Setup: Similar to Scenario 6, designed to cause allocation changes.
  - 🔄 Verification: Assert allocation weights correlate with sentiment strength.

**Implementation Strategy:**
- ✅ Create fixtures in `conftest.py` for reusable components
- ✅ Implement each test scenario in separate test functions
- ✅ Use parameterization for threshold sensitivity tests
- ✅ Next: Implement integration tests and iterate

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
  - ✅ Reddit API collector
    - ✅ Set up Reddit API authentication
    - ✅ Implement subreddit and post search
    - ✅ Add comment extraction and analysis
    - ✅ Implement rate limiting and error handling
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
  - ✅ Implement advanced tokenization (NLTK-based, robust for English)
  - ✅ Add named entity recognition for financial terms (dictionary and regex-based, covers asset symbols, financial terms, prices, cashtags)
  - ✅ Implement text normalization techniques (Unicode normalization, contraction expansion, emoji removal, extensible for slang)
  - ✅ Add support for multiple languages
- ✅ Improve sentiment analysis
  - ✅ Integrate domain-specific sentiment models
  - ✅ Implement fine-tuning on financial text
  - ✅ Add context-aware sentiment analysis
  - ✅ Implement ensemble methods for higher accuracy
- ✅ Add entity recognition
- ✅ Implement company and ticker symbol recognition
- ✅ Add financial metric and event detection
- ✅ Implement relationship extraction between entities
- ✅ Add confidence scoring for entity matching

#### 🔄 Phase 3.6: Advanced Trading Strategy Features (IN PROGRESS)

**Goal:** Enhance the trading strategy capabilities with advanced features for improved performance.

**Detailed Steps:**

1. 🔄 **Implement Advanced Technical Indicators:**
   - 🔄 Add Ichimoku Cloud indicator.
   - 🔄 Add Bollinger Bands with dynamic settings.
   - 🔄 Add Fibonacci retracement levels.
   - 🔄 Add pivot points (standard, Fibonacci, Camarilla, Woodie's).
   - 🔄 Implement volume profile analysis.

2. 🔄 **Create Multi-Timeframe Analysis:**
   - 🔄 Implement hierarchical timeframe analysis.
   - 🔄 Add support for timeframe alignment.
   - 🔄 Create signal confirmation across timeframes.
   - 🔄 Implement timeframe-specific parameter optimization.

3. 🔄 **Add Pattern Recognition:**
   - 🔄 Implement candlestick pattern recognition.
   - 🔄 Add chart pattern detection (head & shoulders, triangles, etc.).
   - 🔄 Create divergence detection for oscillators.
   - 🔄 Implement support/resistance level identification.

4. 🔄 **Enhance Signal Generation:**
   - 🔄 Add signal strength calculation.
   - 🔄 Implement signal confirmation requirements.
   - 🔄 Create signal filtering based on market conditions.
   - 🔄 Add signal expiration and update logic.

#### 🔄 Phase 3.7: Sentiment-Trading Engine Integration (IN PROGRESS)

**Goal:** Integrate sentiment analysis signals into the trading engine with practical considerations for different timeframes and market conditions.

**Detailed Steps:**

1. 🔄 **Implement Sentiment Signal Processor:**
   - 🔄 Create `SentimentSignalProcessor` class in `src/trading/signals/sentiment_processor.py`
   - 🔄 Implement configurable threshold and window size parameters
   - 🔄 Add methods for processing raw sentiment data into trading signals
   - 🔄 Create unit tests with various sentiment scenarios
   - 🔄 Implement signal strength and confidence calculation

2. 🔄 **Develop Timeframe-Aware Integration:**
   - 🔄 Create `TradingModeSelector` class to adapt sentiment usage based on timeframe
   - 🔄 Implement automatic disabling for high-frequency trading (1-5 minute charts)
   - 🔄 Add reduced weighting for intraday timeframes (15-60 minute charts)
   - 🔄 Configure full sentiment integration for swing trading timeframes (4h-1d)
   - 🔄 Add unit tests for each trading mode

3. 🔄 **Implement Market Regime Detection:**
   - 🔄 Create `MarketRegimeDetector` class in `src/trading/market_analysis/`
   - 🔄 Implement algorithms to detect trending, ranging, and volatile market conditions
   - 🔄 Add dynamic sentiment weight adjustment based on market regime
   - 🔄 Create visualization tools for market regime classification
   - 🔄 Add unit tests with historical data from different market regimes

4. 🔄 **Create Sentiment-Based Trading Strategies:**
   - 🔄 Implement `SentimentTrendStrategy` class
   - 🔄 Create `SentimentDivergenceStrategy` class for price-sentiment divergence signals
   - 🔄 Implement `SentimentShockStrategy` for sudden sentiment changes
   - 🔄 Add comprehensive strategy configuration options
   - 🔄 Create unit and integration tests for each strategy

5. 🔄 **Develop Signal Aggregation System:**
   - 🔄 Create `SignalAggregator` class to combine signals from multiple sources
   - 🔄 Implement configurable weighting system for different signal types
   - 🔄 Add conflict resolution logic for contradictory signals
   - 🔄 Implement signal quality tracking over time
   - 🔄 Create visualization tools for signal contribution analysis

6. 🔄 **Extend Backtesting Framework:**
   - 🔄 Modify `Backtester` class to incorporate sentiment data
   - 🔄 Add sentiment data loading to backtesting pipeline
   - 🔄 Implement sentiment-specific performance metrics
   - 🔄 Create visualization tools for sentiment impact analysis
   - 🔄 Add comprehensive tests with historical sentiment and price data

7. 🔄 **Implement Performance Tracking System:**
   - 🔄 Create `SignalPerformanceTracker` class
   - 🔄 Implement metrics for sentiment signal effectiveness
   - 🔄 Add automatic weight adjustment based on historical performance
   - 🔄 Create dashboard components for signal performance visualization
   - 🔄 Implement A/B testing framework for sentiment strategies

### ✅ Phase 4: Genetic Algorithm Optimizer (COMPLETED)
- ✅ Implement parameter optimization framework
  - ✅ Fitness function definition
  - ✅ Population management
  - ✅ Crossover and mutation operations
- ✅ Develop strategy comparison capabilities
  - ✅ Performance metrics calculation
  - ✅ Strategy evaluation
- ✅ Create realistic market condition simulation
  - ✅ Transaction costs
  - ✅ Market biases
{{ ... }}
  - ✅ Slippage modeling

### ✅ Phase 5: Multi-Asset Backtesting Framework (COMPLETED)
- ✅ Implement portfolio-level backtesting
  - ✅ Asset allocation
  - ✅ Correlation analysis
  - ✅ Risk management across the entire portfolio
- ✅ Develop performance metrics for portfolio evaluation
- ✅ Create visualization tools for portfolio performance

### ✅ Phase 6: Modern Dashboard Interface (COMPLETED)

#### Dashboard & Frontend
- ✅ Design and implement a modular dashboard
  - ✅ Audit and map existing dashboard components to required sections
  - ✅ Implement main dashboard layout and navigation (sidebar/tabs)
  - ✅ Add global Mock/Real Data toggle and integrate with all data-fetching components
  - ✅ Ensure Trading Overview section is complete and modular
  - ✅ Ensure Strategy Performance section is complete and modular
  - ✅ Ensure Sentiment Analysis Visualization section is complete and modular
  - ✅ Ensure Portfolio Management section is complete and modular
  - ✅ Standardize UI/UX across all dashboard modules
  - ✅ Add/extend tests for dashboard and data toggling
  - ✅ Document dashboard structure, usage, and extensibility

#### Automated E2E Testing (CI/CD)
- ✅ All dashboard E2E tests are run automatically via GitHub Actions using Cypress in a clean cloud environment.
- ✅ See `.github/workflows/cypress.yml` for the workflow definition.
- ✅ Test results are available in the GitHub Actions tab on every push or pull request.
- ✅ No manual local Cypress troubleshooting is required—CI/CD guarantees reliable, reproducible test results for all contributors.

#### ### Advanced CI/CD & Automation Tasks
- ✅ **Continuous Deployment (CD):**
  - Frontend is automatically deployed via Windsurf/Netlify after tests pass.
  - All build and publish settings in `netlify.toml` are respected.
  - No manual deploy scripts needed for frontend.
  - **Backend CD:** Planned (not yet automated).
- ✅ **Automated Dependency Updates:**
  - Dependabot configured for both frontend (`npm`) and backend (`pip`) dependencies.
  - Updates are automatically tested via CI.
- ✅ **Code Quality & Linting in CI:**
  - ESLint, Prettier, and TypeScript checks are enforced in CI for the frontend.
  - Code style and type safety are robustly maintained.
  - **Backend linting/tests:** Planned.
- ⏳ **Test Coverage Reporting:**
  - Codecov integration is optional and not yet active.
  - Coverage reporting for backend is planned.
- ✅ **Automated Release Notes & Versioning:**
  - `semantic-release` is set up for the frontend for changelogs and version bumps.
  - **Backend:** Not yet automated.

#### Backend API tasks
- ✅ Core REST API endpoints (auth, strategies, some trading endpoints) are implemented and running.
- ⏳ Portfolio, sentiment, and advanced order management endpoints: in progress/planned.
- ✅ WebSocket support is present in the backend codebase.
- ✅ User authentication and authorization are implemented.
- ⏳ Backtesting controls and results API: in progress/planned.

**Status:**
- **Frontend:** 100% complete, tested, and deployed. All dashboard features, technical analysis, E2E testing, and CI/CD (deploy, lint, typecheck, release) are fully automated.
- **Backend:** Core APIs (auth, strategies, basic trading) and WebSocket support are running. Advanced endpoints (portfolio, sentiment, advanced order management, backtesting controls/results) and backend CI/CD automation (auto-deploy, strict lint/coverage enforcement) are still in progress/planned.
- **CI/CD:** Frontend fully automated; backend test/coverage CI is present, but automated deployment and strict enforcement are in progress/planned.
- Dashboard/frontend and E2E testing: **100% complete**
- Backend API (portfolio, sentiment, advanced order management, backtesting) and advanced CI/CD: **in progress/planned**
- Overall Phase 6 completion: **~85–90%**

---

### Crypto Trading Implementation Plan (Twelve Data & Alpha Vantage)

This plan details the implementation of a crypto-focused trading strategy using specific external APIs for data ingestion and analysis.

#### API Selection
*   **Twelve Data:** Use this for real-time crypto prices, live charts, and technical indicators (e.g., RSI, MACD). It’s ideal for the chart analysis agent due to its crypto-specific features and WebSocket support for low-latency updates.
    *   *Details:* Aggregates data from 180+ exchanges, offers extensive indicators.
    *   *Cost:* Starts at $29/month (free tier available with limits).
*   **Alpha Vantage:** Use this for sentiment analysis via its News Sentiment API, powering the sentiment agent. It also provides additional crypto data if needed.
    *   *Details:* Offers real-time sentiment data from financial news, key for gauging market mood.
    *   *Cost:* Free tier (750 calls/month, no intraday data) or $49.99/month for premium access.

#### Integration Strategy
*   **Real-Time Data (Twelve Data):** Implement WebSocket connections (`wss://ws.twelvedata.com`) to stream live crypto prices and chart updates. This ensures the chart analysis agent has the freshest data.
*   **Sentiment Data (Alpha Vantage):** Use REST APIs to fetch sentiment data periodically (e.g., every 5-15 minutes, depending on trading frequency). Sentiment changes slower than price data, so real-time streaming isn’t critical.

#### Development Steps
Follow these steps to build and integrate the system:
1.  **API Key Acquisition**
    *   Sign up for Twelve Data and Alpha Vantage to obtain API keys.
    *   Store keys securely using environment variables or a secrets manager.
2.  **Data Ingestion**
    *   **Twelve Data (WebSocket):**
        *   Implement a WebSocket client to connect to `wss://ws.twelvedata.com`.
        *   Subscribe to relevant crypto pairs (e.g., BTC/USD, ETH/USD).
        *   Handle real-time price updates and indicator calculations.
    *   **Alpha Vantage (REST):**
        *   Set up a REST client to call the News Sentiment API.
        *   Schedule periodic requests (e.g., every 15 minutes) to fetch sentiment scores for key crypto assets.
3.  **Data Processing**
    *   **Chart Analysis:**
        *   Parse real-time data from Twelve Data for live charting.
        *   Compute technical indicators (e.g., moving averages, RSI) using built-in API functions or a library like TA-Lib.
    *   **Sentiment Analysis:**
        *   Extract sentiment scores from Alpha Vantage’s API responses.
        *   Aggregate sentiment data over time to identify trends (e.g., bullish or bearish shifts).
4.  **Agent Development**
    *   **Chart Analysis Agent:**
        *   Use Twelve Data’s real-time feeds to monitor price movements and trigger alerts based on technical indicators (e.g., crossovers, breakouts).
        *   Implement logic for generating buy/sell signals based on indicator thresholds.
    *   **Sentiment Agent:**
        *   Use Alpha Vantage’s sentiment data to assess market mood.
        *   Combine sentiment scores with technical signals to refine trading decisions (e.g., avoid buying during negative sentiment spikes).
5.  **System Integration**
    *   Combine outputs from both agents to make informed trading decisions.
    *   Define rules for how sentiment and technical signals interact (e.g., require both to align for a trade).
    *   Ensure the system handles asynchronous data updates (real-time prices vs. periodic sentiment).

#### Key Considerations
*   **API Limits:**
    *   Monitor usage to avoid hitting rate limits (e.g., Alpha Vantage’s free tier has 750 calls/month).
    *   Upgrade to premium plans if higher frequency or intraday data is needed.
*   **Data Consistency:**
    *   Align timestamps between Twelve Data and Alpha Vantage to avoid mismatched signals.
*   **Error Handling:**
    *   Add retry logic for API failures.
    *   Set up alerts for downtime or degraded performance.
*   **Scalability:**
    *   Ensure the system can handle multiple crypto pairs if expanded.
    *   Batch sentiment requests to optimize API usage.

#### Testing
*   **Historical Data Simulation:**
    *   Use historical data from Twelve Data to backtest the chart analysis agent’s signals.
    *   Validate sentiment analysis by comparing past sentiment scores with market movements.
*   **Live Testing:**
    *   Run the system in a paper trading environment to simulate real trades without financial risk.
    *   Check for discrepancies between live and historical performance.

#### Deployment
*   **Monitoring:**
    *   Set up logging for API response times and system health.
    *   Use tools like Prometheus or Grafana to track performance metrics.
*   **Failover Mechanisms:**
    *   Implement fallback logic (e.g., switch to cached data if an API fails).
    *   Schedule regular health checks for API connectivity.

---

### ⏳ Phase 7: Integration and Deployment (PENDING)
**Goal:** Create a fully integrated system that combines all components and deploy it to production environments.

#### 7.1 System Integration

1. ⏳ **Trading Agent Service Integration**
   - ⏳ Connect backend API with trading agent core
   - ⏳ Add request/response mapping between API models and core domain models
   - ⏳ Implement proper error handling and logging
   - ⏳ Create integration tests for API-to-agent flow

2. ⏳ **Live Trading Bridge**
   - ⏳ Implement bridge between backtesting and live trading
   - ⏳ Create unified interface for market/limit/stop order placement
   - ⏳ Add support for both paper trading and real trading modes
   - ⏳ Implement proper trading safeguards (max position size, max loss, etc.)
   - ⏳ Create thorough testing in paper trading mode

3. ⏳ **Data Pipeline Integration**
   - ⏳ Connect sentiment analysis system with data acquisition module
   - ⏳ Set up scheduled tasks for data collection and processing
   - ⏳ Implement caching layer for performance optimization
   - ⏳ Add data validation and error recovery mechanisms

4. ⏳ **Dashboard-Backend Integration**
   - ⏳ Finalize WebSocket implementation for real-time updates
   - ⏳ Complete authentication and authorization flow
   - ⏳ Add proper error handling and status reporting
   - ⏳ Create end-to-end tests for dashboard-backend communication

#### 7.2 Containerization

1. ⏳ **Docker Containerization**
   - ⏳ Create Dockerfile for backend services
   - ⏳ Create Dockerfile for frontend application
   - ⏳ Add multi-stage builds for optimization
   - ⏳ Implement proper environment variable management
   - ⏳ Setup Docker Compose for local development

2. ⏳ **Container Orchestration**
   - ⏳ Configure Kubernetes deployment manifests
   - ⏳ Set up health checks and readiness probes
   - ⏳ Implement auto-scaling configuration
   - ⏳ Configure resource limits and requests
   - ⏳ Add persistent volume claims for data storage

#### 7.3 CI/CD Pipeline Completion

1. ⏳ **Backend CI/CD Implementation**
   - ⏳ Set up automated testing in CI
   - ⏳ Implement code quality checks (linting, type checking)
   - ⏳ Add test coverage reporting
   - ⏳ Create automated deployment workflow
   - ⏳ Configure staging and production environments
   - ✅ Integrate automated security testing with OWASP ZAP in CI pipeline

2. ⏳ **Monitoring and Observability**
   - ⏳ Implement application metrics collection
   - ⏳ Set up centralized logging
   - ⏳ Add distributed tracing for request flow
   - ⏳ Create dashboards for system monitoring
   - ⏳ Configure alerts for critical conditions

#### 7.4 Security and Compliance

1. 🔄 **Security Measures**
   - ✅ Implement robust Content Security Policy (CSP) with reporting 
   - ✅ Add comprehensive security audit logging system
   - ✅ Implement automated security testing with OWASP ZAP
   - ⏳ Add API rate limiting
   - ⏳ Add API key rotation mechanism
   - ⏳ Set up secure storage for credentials
   - ⏳ Perform security audit and penetration testing
   - ⏳ Implement proper CORS policy

2. ⏳ **Data Protection**
   - ⏳ Ensure GDPR compliance for user data
   - ⏳ Implement data encryption at rest and in transit
   - ⏳ Add data backup and recovery procedures
   - ⏳ Create data retention and purging policies

#### 7.5 Documentation and Handover

1. ⏳ **System Documentation**
   - ⏳ Create deployment guides for different environments
   - ⏳ Document system architecture and component interactions
   - ⏳ Add troubleshooting guides and FAQs
   - ⏳ Create API documentation with examples

2. ⏳ **User Documentation**
   - ⏳ Create user guides for the dashboard
   - ⏳ Add documentation for strategy configuration
   - ⏳ Create trading and backtesting tutorials
   - ⏳ Document risk management features

### ✅ Phase 8: Continuous Improvement (IN PROGRESS)
- ✅ Performance optimization
  - ✅ Implement memoization and caching strategies
  - ✅ Create batch processing for API calls
  - ✅ Add performance metrics tracking
  - ✅ Create performance monitoring dashboard
  - ✅ Implement debounce and throttle utilities
  - ✅ Add performance testing utilities and benchmarking
- ⏳ Additional trading strategies
- ⏳ Enhanced visualization features
- ⏳ User feedback integration
