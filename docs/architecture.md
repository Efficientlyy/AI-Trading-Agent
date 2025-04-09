# AI Trading Agent Architecture

## 1. Overview

### 1.1. Project Goal

The primary goal of this project is to develop an advanced, AI-powered cryptocurrency trading system designed for **sustained profitability**. It aims to navigate volatile market conditions through:

*   **Adaptive Strategies:** Employing sophisticated machine learning techniques that adjust to changing market dynamics.
*   **Robust Risk Management:** Incorporating confidence scoring and dynamic adjustments based on prediction certainty.
*   **Continuous Optimization:** Utilizing methods like Genetic Algorithms and online learning to refine strategies over time.
*   **High Performance:** Leveraging Rust for performance-critical computations.

### 1.2. Core Principles

*   **Modularity:** Components are designed to be independent and interchangeable.
*   **Testability:** Emphasis on unit and integration testing for reliability.
*   **Reusability:** Core components can be leveraged across different strategies and modules.
*   **Extensibility:** The architecture allows for easy addition of new data sources, indicators, strategies, and models.
*   **Performance:** Critical sections are optimized, including planned Rust integration.
*   **Adaptability:** The system is built to react to detected changes in market regimes and model performance (concept drift).

### 1.3. Target Audience

This document is primarily intended for developers and contributors involved in the design, implementation, and maintenance of the AI Trading Agent system.

## 2. High-Level Architecture

The system is composed of several interconnected layers and components:

```mermaid
graph TD
    subgraph User Interface (Future)
        DASH[Dashboard (Flask/Dash)]
    end

    subgraph Core Logic
        DATA_ACQ[Data Acquisition] --> DATA_PROC
        DATA_PROC[Data Processing (Python/Rust)] --> MRD
        DATA_PROC --> STRAT
        MRD[Market Regime Detection (Ensemble + Confidence)] --> STRAT
        SENT[Sentiment Analysis System] --> MRD --> STRAT
        STRAT[Strategy Layer (Adaptive ML / Rules)] --> OPT
        STRAT --> TE
        OPT[Optimization (Genetic Algo - Offline)] --> STRAT
        TE[Trading Engine (Orders, Positions)] --> EXEC
        TE --> PM
        PM[Portfolio Management] --> STRAT
        EXEC[Execution Layer (Sim/Live)] --> TE
        BACKTEST[Backtesting Framework (Python/Rust)] --> DATA_ACQ
        BACKTEST --> DATA_PROC
        BACKTEST --> MRD
        BACKTEST --> STRAT
        BACKTEST --> TE
        BACKTEST --> PM
        BACKTEST --> METRICS[Performance Metrics]
    end

    subgraph Infrastructure
        CONFIG[Configuration (YAML)]
        LOG[Logging]
        RUST[Rust Extensions (PyO3)]
    end

    CONFIG --> DATA_ACQ
    CONFIG --> DATA_PROC
    CONFIG --> MRD
    CONFIG --> SENT
    CONFIG --> STRAT
    CONFIG --> OPT
    CONFIG --> TE
    CONFIG --> EXEC
    CONFIG --> BACKTEST
    CONFIG --> LOG
    CONFIG --> RUST

    DATA_PROC -.-> RUST
    BACKTEST -.-> RUST

    DASH --> STRAT
    DASH --> TE
    DASH --> PM
    DASH --> BACKTEST
    DASH --> LOG
    DASH --> OPT
    DASH --> EXEC
```

## 3. Component Breakdown

### 3.1. Data Acquisition
*   **Purpose:** Fetches historical and real-time market data (prices, volume, etc.).
*   **Components:**
    *   `BaseDataProvider`: Interface defining data fetching methods.
    *   `CcxtProvider`: Fetches data from various exchanges via CCXT.
    *   `MockDataProvider`: Provides static or generated data for testing.
    *   `DataService`: Orchestrates provider selection based on configuration.
*   **Notes:** Potential for Rust integration for high-frequency data handling if needed.

### 3.2. Data Processing
*   **Purpose:** Cleans raw data, calculates technical indicators, and generates features for strategies and models.
*   **Components:**
    *   `indicators.py`: Functions for calculating TA indicators (SMA, EMA, RSI, MACD, etc.).
    *   `feature_engineering.py`: Pipeline for creating derived features.
*   **Notes:** **Planned Rust Integration** (`rust_extensions/`) using `ta-rs`, `ndarray`, and PyO3 for significant performance improvements in indicator calculation.

### 3.3. Market Regime Detection
*   **Purpose:** Identifies the current market state (e.g., Bull Trend, Bear Trend, Volatile Range, Low Volatility) to enable strategy adaptation. This is a *core intelligent component*.
*   **Method:** Utilizes an **Ensemble** approach, combining signals from multiple detector types:
    *   Volatility-based detectors
    *   Trend-based detectors
    *   Momentum-based detectors
    *   Clustering-based detectors
    *   HMM-based detectors
    *   **Sentiment Analysis** (scores fed as input)
*   **Confidence Scoring:** Calculates a reliability score for the regime prediction based on detector agreement, historical accuracy, data quality, etc. This score informs risk management and strategy execution.
*   **Components:** Detector implementations, Ensemble Manager, Confidence Calculator.

### 3.4. Sentiment Analysis System
*   **Purpose:** Gathers and processes data from external sources (e.g., Twitter, news headlines) to gauge market sentiment.
*   **Method:** Involves data scraping/APIs, Natural Language Processing (NLP) for text cleaning and analysis, and sentiment scoring algorithms.
*   **Integration:** Feeds sentiment scores into the Market Regime Detection ensemble and can potentially be used directly by specific strategies.
*   **Components:** Data Collectors, NLP Pipeline, Sentiment Scorer.

### 3.5. Strategy Layer
*   **Purpose:** Encapsulates the decision-making logic for generating trading signals (Buy, Sell, Hold).
*   **Method:**
    *   `BaseStrategy` Interface: Defines standard methods (`generate_signals`).
    *   Implementations: Can range from simple technical rule-based strategies (e.g., MA Crossover) to complex **Machine Learning models** (Random Forest, Gradient Boosting planned).
    *   **Regime Adaptation:** Strategies explicitly use the detected market regime and confidence score to modify their behavior (e.g., change parameters, use different models, adjust position sizing, halt trading).
    *   **Online Learning:** Framework planned to allow ML models to adapt to concept drift over time.
*   **Components:** `BaseStrategy`, specific strategy classes, ML model wrappers, signal generation logic.

### 3.6. Genetic Algorithm Optimizer
*   **Purpose:** Performs offline optimization of strategy parameters to maximize performance based on backtesting results.
*   **Method:** Uses evolutionary algorithms to search the parameter space (e.g., indicator periods, sentiment thresholds, risk management settings). Fitness is evaluated using backtesting performance metrics.
*   **Components:** GA Engine, Parameter Space Definition, Fitness Function (interfacing with Backtesting).

### 3.7. Trading Engine
*   **Purpose:** Manages the lifecycle of orders, maintains the state of open positions, and tracks portfolio value and cash balance.
*   **Components:**
    *   Core Data Models (`Order`, `Trade`, `Position`, `Portfolio` using Pydantic).
    *   `OrderManager`: Handles order submission logic, status updates, and fills processing.
    *   `ExecutionHandler` (Planned): Simulates or executes trades via exchange APIs.
    *   `PortfolioManager` (Potential): May evolve for more complex multi-asset portfolio logic.

### 3.8. Backtesting Framework
*   **Purpose:** Simulates the execution of strategies on historical data to evaluate performance before deployment.
*   **Components:**
    *   `Backtester`: Main event loop orchestrating data feeding, signal generation, and trade simulation. **Planned Rust Integration** for core loop acceleration.
    *   `PerformanceMetrics`: Calculates key metrics (Sharpe Ratio, Sortino Ratio, Max Drawdown, Win Rate, PnL, etc.).
*   **Notes:** Supports walk-forward analysis and integrates with the GA Optimizer.

### 3.9. Portfolio Management
*   **Purpose:** Manages overall risk and capital allocation, potentially across multiple assets and strategies (Focus of Phase 5).
*   **Components:** Currently integrated within `Portfolio` model; may expand to a dedicated manager for multi-asset scenarios.

### 3.10. Execution Layer
*   **Purpose:** Handles communication with cryptocurrency exchanges for placing orders and receiving trade confirmations.
*   **Components:** `ExecutionHandler` interface with implementations for:
    *   Simulation (using Trading Engine logic).
    *   Live Trading (via exchange APIs, e.g., CCXT - Future).

### 3.11. Configuration
*   **Purpose:** Manages all system settings, parameters, and API keys.
*   **Method:** Primarily uses YAML files (`config/config.yaml`) for easy editing and version control.

### 3.12. Logging
*   **Purpose:** Records system events, trades, errors, and debug information.
*   **Method:** Uses Python's standard `logging` module configured for structured output.

### 3.13. Rust Extensions (`rust_extensions/`)
*   **Purpose:** Provides high-performance implementations for computationally intensive tasks.
*   **Method:** Uses PyO3/Maturin to create Python bindings for Rust code.
*   **Planned Integrations:**
    *   Technical indicator calculations (Data Processing).
    *   Core backtesting event loop (Backtesting Framework).

### 3.14. Dashboard UI (Future)
*   **Purpose:** Provides a web-based interface for monitoring system health, visualizing performance, managing strategies, and potentially manual control overrides.
*   **Technology:** Planned using Flask/Dash.

## 4. Decision Flow Example (Backtesting)

1.  **Load Data:** `Backtester` requests historical data from `Data Acquisition`.
2.  **Process Data:** Data is passed to `Data Processing` (using Python/Rust) to calculate indicators and features.
3.  **Detect Regime:** Processed data (and potentially sentiment data) is fed into `Market Regime Detection` to get the current regime and confidence score.
4.  **Generate Signal:** The active `Strategy` receives processed data, regime, and confidence, then generates a trading signal (Buy/Sell/Hold). ML strategies use their internal models, potentially adapted for the regime.
5.  **Simulate Execution:** `Backtester` sends the signal/order to the `Trading Engine` (via a simulated `ExecutionHandler`), which updates `Position` and `Portfolio` based on simulated fills.
6.  **Calculate Metrics:** Loop continues; `PerformanceMetrics` are updated periodically or at the end.
7.  **(Offline Optimization):** `Genetic Algorithm Optimizer` runs multiple backtests, adjusting strategy parameters based on `PerformanceMetrics` to find optimal configurations.

## 5. Technology Stack

*   **Core Language:** Python 3.11+
*   **Performance:** Rust (with PyO3/Maturin bindings)
*   **Data Handling:** Pandas, NumPy, Pydantic
*   **Machine Learning:** Scikit-learn, potentially others (TensorFlow/PyTorch if Deep Learning is explored)
*   **Data Acquisition:** CCXT, potentially yfinance
*   **Testing:** Pytest
*   **Configuration:** PyYAML
*   **Web UI (Future):** Flask, Plotly Dash

## 6. Key Design Decisions

*   **Regime-Awareness:** Centrality of the Market Regime Detection ensemble for adaptability.
*   **Confidence Scoring:** Explicitly modeling prediction certainty for risk management.
*   **Ensemble Methods:** Using multiple models/detectors for robustness.
*   **ML-Centric:** Leveraging ML for complex pattern recognition and adaptation (including online learning).
*   **Sentiment Integration:** Incorporating alternative data sources for a more holistic market view.
*   **Performance Focus:** Strategic use of Rust for identified bottlenecks.
*   **Modularity:** Enabling independent development and testing of components.
*   **Optimization:** Dedicated GA framework for refining strategies.

## 7. Continuous Improvement

The system is designed for ongoing enhancement. This includes:
*   Monitoring performance metrics via the dashboard.
*   Using the GA to periodically re-optimize strategies.
*   Allowing ML models to adapt through online learning.
*   Analyzing logs and backtest results to identify areas for improvement.
*   Facilitating human-in-the-loop analysis and adjustments via the dashboard interface.

---

## Genetic Algorithm Optimizer

The GA optimizer automates tuning of strategy parameters to maximize performance.

### Overview

- Uses evolutionary algorithms to search parameter space
- Evaluates each candidate via backtesting
- Selects, crosses over, and mutates candidates over generations
- Returns the best parameter set found

### Defining Parameter Spaces

Provide a dictionary mapping parameter names to lists of possible values:

```python
param_space = {
    "fast_period": list(range(5, 30)),
    "slow_period": list(range(20, 100)),
    "threshold": [0.0, 0.01, 0.02, 0.05]
}
```

### Writing Fitness Functions

Create a function that:

- Accepts a parameter dictionary
- Runs a backtest with those parameters
- Returns a performance metric (e.g., Sharpe ratio)

Example:

```python
def run_backtest_with_params(params):
    # Initialize strategy with params
    # Run backtest
    # Return Sharpe ratio
    return sharpe_ratio
```

### Running Optimization

Use the `GAOptimizer` class:

```python
optimizer = GAOptimizer(
    param_space=param_space,
    fitness_func=run_backtest_with_params,
    population_size=20,
    generations=50,
    crossover_rate=0.7,
    mutation_rate=0.1
)
best_params = optimizer.evolve()
```

### Analyzing Results

- Run multiple experiments for robustness
- Save results to JSON or CSV
- Analyze parameter distributions and performance metrics
- Select best parameters for deployment or further testing

---
---

## Exception Hierarchy

- `TradingEngineError`
  - `OrderValidationError`
  - `ExecutionError`
  - `PortfolioUpdateError`
  - `DataProviderError`

Use these exceptions for clear, structured error handling.

---

## Contribution Guidelines

- Follow **PEP8** and use **type annotations**
- Write **docstrings** for all public classes and methods
- Use **structured logging** with context
- Raise **custom exceptions** instead of generic ones
- Add **unit and integration tests** for new features
- Update **docs/PLAN.md** and this architecture doc with changes
- Use **feature branches** and submit **pull requests** for review
- Run tests and linters before submitting code

---
