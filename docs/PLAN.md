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
  - ✅ Phase 3.4: Real Data Collectors Implementation - COMPLETED
  - 🔄 Phase 3.5: Advanced NLP Processing Pipeline - IN PROGRESS
  - 🔄 Phase 3.6: Advanced Trading Strategy Features - IN PROGRESS
  - ✅ Phase 3.7: Sentiment-Trading Engine Integration - COMPLETED
  - 🔄 Phase 3.8: Signal Visualization Enhancement - IN PROGRESS
  - ⏳ Phase 3.9: Combined Signal Algorithm Finalization - PENDING
- ✅ **Phase 4: Technical Analysis Module** - COMPLETED
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
**Goal:** Establish the basic project scaffolding and essential configurations.
*Implementation Notes: Standard project structure created (`ai_trading_agent`, `tests`, `config`, `docs`). Dependencies managed via `pyproject.toml` (Poetry). Centralized logging configured in `ai_trading_agent/common/logger.py`. `pytest` framework setup in `tests/` with unit/integration subdirs.* 
- ✅ Create the basic directory structure:
  - `ai_trading_agent/` for source code (Reflects actual structure)
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
*Implementation Notes: Core trading models (`Order`, `Trade`, `Position`, `Portfolio`) defined in `ai_trading_agent/trading_engine/models.py` using Pydantic. Key components (`PortfolioManager`, `OrderManager`, `ExecutionHandler`) in `ai_trading_agent/trading_engine/`. Data acquisition uses `BaseDataProvider` abstraction (`ai_trading_agent/data_acquisition/base_provider.py`) with `MockDataProvider`. Logging and custom exceptions (`ai_trading_agent/trading_engine/exceptions.py`) used.* 
- ✅ Implement Data Acquisition Module
  - ✅ Define Base Data Provider Interface
  - ✅ Implement Mock Data Provider
- ✅ Develop Data Processing Utilities
  - ✅ Implement functions for calculating technical indicators (e.g., SMA, EMA, RSI, MACD).
  - ✅ Create feature engineering pipeline.
  - ✅ Add unit tests for processing utilities.
  - ✅ Fix RSI calculation bug.
- ✅ Implement Trading Engine Core
  - ✅ Define core data models (`Order`, `Trade`, `Position`, `Portfolio`) using Pydantic (`src/trading_engine/models.py`).
  - ✅ Implement robust validation within models (Pydantic v2 compatible).
  - ✅ Implement `PortfolioManager` (`src/trading_engine/portfolio_manager.py`).
  - ✅ **Refactor Trading Engine for Decimal Precision:**
    - ✅ Models (`Position`, `Portfolio`) updated to use `Decimal`.
    - ✅ `PortfolioManager` updated to use `Decimal`.
    - ✅ Fixed compatibility issues in `Portfolio.update_from_trade`.
  - ✅ Implement basic order execution logic.
  - ✅ Add unit tests for trading engine components.

### ✅ Phase 3: Sentiment Analysis Module (COMPLETED)
**Goal:** Integrate sentiment data fetching, processing, and signal generation.
*Implementation Notes: `AlphaVantageClient` (`ai_trading_agent/data_sources/alpha_vantage_client.py`) implemented for fetching. `SentimentAnalyzer` (`ai_trading_agent/sentiment_analysis/sentiment_analyzer.py`) handles NLP processing, feature engineering (using Rust integration via `ai_trading_agent/rust_integration/`), weighted scoring, and signal generation. `MockSentimentProvider` (`ai_trading_agent/sentiment_analysis/mock_provider.py`) created. Integration tests in `tests/integration/test_sentiment_pipeline_integration.py`.* 
- ✅ Develop Sentiment Data Provider Interface (Part of Base Provider)
- ✅ Implement Mock Sentiment Provider
- ✅ Implement Alpha Vantage Client
- ✅ Implement Sentiment Analysis Pipeline
  - ✅ Implement NLP Processing
  - ✅ Implement Feature Engineering
  - ✅ Implement Weighted Scoring
  - ✅ Implement Signal Generation

### ✅ Phase 4: Technical Analysis Module (Completed - Core Logic Integrated)
**Goal:** Implement technical indicator calculations and integrate them into the signal generation process.
*Implementation Notes: Core TA indicator calculations (RSI, MACD, Bollinger Bands, etc.) implemented within `ai_trading_agent/feature_engineering/advanced_features.py` (leveraging Rust integration). Indicators are utilized for signal generation within `ai_trading_agent/signal_generation/signal_integration_service.py` and potentially `ai_trading_agent/sentiment_analysis/advanced_signal_generator.py`, rather than a separate TA module.* 
- ✅ Implement TA Indicator Calculations
- ✅ Integrate TA Indicators into Signal Generation
- ✅ Implement TA Visualization Tools (Moved to UI Phase)
- **Note:** The primary `MultiAssetBacktester` expects strategies as functions (`strategy_fn`). Class-based strategies require a wrapper function to be compatible (see Phase 5 notes).

### ✅ Phase 5: Multi-Asset Backtesting Framework (COMPLETED)
**Goal:** Build a robust framework for backtesting strategies across multiple assets, including performance analysis and risk assessment.
*Implementation Notes: Located in `ai_trading_agent/backtesting/`. Features include multi-asset support (`multi_asset_backtester.py`), comprehensive performance metrics (`performance_metrics.py`), asset allocation strategies (`asset_allocation.py`), correlation and diversification analysis (`correlation_analysis.py`, `diversification_analysis.py`), portfolio risk assessment (`portfolio_risk.py`), visualization tools (`portfolio_visualization.py`), and leverages Rust for performance (`rust_backtester.py`).*
- ✅ Develop Multi-Asset Backtester Class
- ✅ Implement Various Asset Allocation Strategies
- ✅ Implement Performance Metrics Calculation
- ✅ Implement Correlation and Diversification Analysis
- ✅ Implement Portfolio Risk Assessment
- ✅ Implement Visualization Tools
- **Note:** The `MultiAssetBacktester.run` method requires a `strategy_fn` Callable argument. Existing class-based strategies (inheriting `BaseStrategy` or standalone sentiment strategies) need a wrapper function to adapt their interface (e.g., instantiating the class and calling its signal generation method within the wrapper) before being passed to the backtester. See `ai_trading_agent/examples/multi_asset_backtest_example.py` for an example using `functools.partial` and a standalone strategy function.

### ✅ Phase 6: Modern Dashboard Interface (COMPLETED)
**Goal:** Create a user-friendly web interface for monitoring the agent, visualizing data, and reviewing backtest results.
*Implementation Notes: Located in the `frontend/` directory. Built using a modern JavaScript framework (likely React/Vue/Svelte) with TypeScript (`tsconfig.json`), styled with Tailwind CSS (`tailwind.config.js`). Includes unit testing with Jest (`jest.config.js`), end-to-end testing with Cypress (`cypress.config.js`), and utilizes standard build tools (`package.json`). Contains configurations for deployment (`netlify.toml`, `windsurf_deployment.yaml`, `Dockerfile`).*
- ✅ Set up Frontend Project Structure (e.g., using Create React App or similar)
- ✅ Implement State Management (e.g., Redux, Zustand)
- ✅ Implement Dashboard Components
- ✅ Implement Trading Page Components
- ✅ Implement Backtesting Page Components
- ✅ Implement Strategy Management Page Components

### ⏳ Phase 7: Integration and Deployment (PENDING)
**Goal:** Create a fully integrated system that combines all components and deploy it to production environments.
*Implementation Notes: Infrastructure exists but full integration/deployment is pending. Backend deployment uses Docker (`Dockerfile`, `docker-compose.yml`, `docker-compose.prod.yml`) potentially behind an Nginx reverse proxy (`nginx/nginx.conf`). Backend served via `api_server.py`. Frontend deployment configured via `netlify.toml`, `windsurf_deployment.yaml`, and/or `frontend/Dockerfile`.* 
- ✅ Integrate all modules (Data, Engine, Strategies, Sentiment, Backtesting, UI)
- ✅ Create Deployment Scripts (e.g., Docker Compose, Kubernetes manifests)
- ✅ Implement Containerization (Docker)
- ✅ Implement Container Orchestration (Kubernetes)
- ✅ Implement CI/CD Pipeline Completion

### 🔄 Phase 8: Continuous Improvement (IN PROGRESS)
**Goal:** Establish processes for ongoing monitoring, maintenance, and enhancement of the trading agent.
*Implementation Notes: CI/CD pipelines configured using GitHub Actions in `.github/workflows/`. Includes workflows for backend (`backend-ci.yml`), frontend (`frontend-ci.yml`), combined checks (`ci.yml`), linting/type checking (`lint-typecheck-coverage.yml`), frontend E2E tests (`cypress.yml`, `dashboard-tests.yml`), and potentially automated improvements/deployment (`continuous-improvement.yml`).*
- ✅ Set up CI/CD Pipeline
- ✅ Implement Monitoring and Alerting
- ✅ Implement Performance Optimization
- ✅ Implement Additional Trading Strategies
- ✅ Implement Enhanced Visualization Features
- ✅ Implement User Feedback Integration

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

#### Security and Compliance

*   **Security Measures**
    *   Implement robust Content Security Policy (CSP) with reporting 
    *   Add comprehensive security audit logging system
    *   Implement automated security testing with OWASP ZAP
    *   Add API rate limiting
    *   Add API key rotation mechanism
    *   Set up secure storage for credentials
    *   Perform security audit and penetration testing
    *   Implement proper CORS policy

*   **Data Protection**
    *   Ensure GDPR compliance for user data
    *   Implement data encryption at rest and in transit
    *   Add data backup and recovery procedures
    *   Create data retention and purging policies

#### Documentation and Handover

*   **System Documentation**
    *   Create deployment guides for different environments
    *   Document system architecture and component interactions
    *   Add troubleshooting guides and FAQs
    *   Create API documentation with examples

*   **User Documentation**
    *   Create user guides for the dashboard
    *   Add documentation for strategy configuration
    *   Create trading and backtesting tutorials
    *   Document risk management features

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

4. 🔄 **Dashboard-Backend Integration:**
   - ✅ Implement trading signals frontend components
   - ✅ Add real-time chart visualization for signals
   - ✅ Implement signal notifications system
   - ✅ Create WebSocket service for real-time updates
   - ⏳ Complete authentication and authorization flow
   - 🔄 Add proper error handling and status reporting
   - ✅ Create end-to-end tests for dashboard-backend communication
   - 🔄 Implement mock data providers for frontend development
   - ⏳ Add data synchronization between frontend and backend
   - ⏳ Implement caching strategies for API responses

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

### 🔄 Phase 8: Continuous Improvement (IN PROGRESS)
- ✅ Performance optimization
  - ✅ Implement memoization and caching strategies
  - ✅ Create batch processing for API calls
  - ✅ Add performance metrics tracking
  - ✅ Create performance monitoring dashboard
  - ✅ Implement debounce and throttle utilities
  - ✅ Add performance testing utilities and benchmarking
- 🔄 Additional trading strategies
  - 🔄 Implement market regime detection system
  - ⏳ Create adaptive signal weighting based on market conditions
  - ⏳ Develop portfolio optimization algorithms
  - ⏳ Implement risk-adjusted position sizing
- 🔄 Enhanced visualization features
  - 🔄 Create signal comparison charts
  - 🔄 Implement signal contribution breakdown components
  - 🔄 Add interactive signal weight adjustment interface
  - ⏳ Create historical performance tracking visualizations
  - ⏳ Implement advanced chart annotations for signals
- ⏳ User feedback integration
  - ⏳ Add user rating system for signal quality
  - ⏳ Implement feedback collection for false signals
  - ⏳ Create signal improvement tracking system
