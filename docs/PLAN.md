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
    *   [ ] Implement `ExecutionHandler` to simulate trade execution (`src/trading_engine/execution_handler.py` - previously `execution_simulator.py`).
    *   [ ] Implement `PortfolioManager` logic (partially covered by `Portfolio` model methods, may need a dedicated manager later).
    *   [x] Add unit tests for OrderManager (`tests/unit/trading_engine/test_order_manager.py`).
    *   [ ] Add unit tests for other managers and handlers.
4.  **Develop Backtesting Framework:**
    *   [ ] Implement basic backtesting loop (`src/backtesting/backtester.py`).
    *   [ ] Define performance metrics calculation (`src/backtesting/performance_metrics.py`).
5.  **Define Base Strategy Interface:**
    *   [ ] Create `BaseStrategy` class with methods for initialization and generating signals (`generate_signals`).
    *   [ ] Add basic example strategy implementation.

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
- Create interactive components
  - Strategy parameter adjustment
  - Backtesting controls
  - Real-time monitoring
- Implement authentication and security features

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