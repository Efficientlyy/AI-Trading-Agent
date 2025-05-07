# AI Trading Agent: Signal Integration & Multi-Agent Communication Plan

## 1. Objective

Implement a robust signal aggregation and communication system between specialized trading agents (Strategies) as envisioned in `PLAN.md`. The goal is to enable effective collaboration, allowing a central "Decision Agent" component (Strategy Manager) to combine insights from multiple sources before generating final trading signals.

## 2. Current State & Problem

*   **Lack of Combination:** Existing `StrategyManager` implementations (`SimpleStrategyManager`, `SentimentStrategyManager`) do not combine signals from multiple strategies. `SimpleStrategyManager` uses only the first strategy; others implement agent-specific logic.
*   **Structural Inconsistencies:** Code scanning revealed:
    *   Two different `BaseStrategy` definitions (`agent/strategy.py` and `strategies/base_strategy.py`) with differing interfaces (signal dict vs. order list).
    *   Multiple, potentially conflicting `SentimentStrategy` definitions/implementations (`agent/strategy.py` vs. `sentiment_analysis/strategy.py`).
    *   Generic risk/utility functions located within `sentiment_analysis/strategy.py`.

## 3. Phase 1.5 (Prerequisite): Structural Refactoring

**Goal:** Establish a clean, consistent foundation before building the integration logic.

*   **3.1. Standardize on `agent/strategy.py::BaseStrategy`:**
    *   **Action:** Refactor `MACrossoverStrategy` and any other strategies using `strategies/base_strategy.py` to inherit from `agent/strategy.py::BaseStrategy`.
    *   **Action:** Modify their `generate_signals` methods to return `Dict[str, int]` (for now) instead of `List[Order]`. Remove internal order/sizing logic.
    *   **Action:** Deprecate/remove `strategies/base_strategy.py`.
*   **3.2. Consolidate Sentiment Strategy:**
    *   **Action:** Identify the canonical `SentimentStrategy` logic (likely in `agent/strategy.py`).
    *   **Action:** Move generic risk/utility functions (Kelly, ATR stops, risk parity etc.) from `sentiment_analysis/strategy.py` to a dedicated `ai_trading_agent/risk` or `ai_trading_agent/utils` module.
    *   **Action:** Ensure the canonical `SentimentStrategy` inherits from the canonical `BaseStrategy`.
    *   **Action:** Deprecate/remove the abstract/dummy strategies in `sentiment_analysis/strategy.py`.
*   **3.3. Verify Consistency:** Ensure all specialized strategies inherit from `agent/strategy.py::BaseStrategy` and have a basic `generate_signals -> Dict[str, int]` implementation.

## 4. Phase 2 (Post-Refactoring): Design Advanced Communication Framework

**Goal:** Define the interfaces, data structures, and core components for agent collaboration.

*   **4.1. Define Standardized Agent Output:**
    *   **Action:** Modify `BaseStrategy.generate_signals` (the canonical one) to return a richer dictionary per symbol, enabling advanced combination. Proposal:
      ```python
      # Output for a single symbol from one agent:
      {
          'signal_strength': float, # -1.0 to +1.0
          'confidence_score': float, # 0.0 to 1.0
          'signal_type': str, # Optional: 'technical', 'sentiment', etc.
          'metadata': dict # Optional: {'indicator': 'RSI_14', ...}
      }
      # Method signature changes, e.g., -> Dict[str, Dict[str, Any]]
      ```
    *   **Action:** Update all concrete strategy implementations to conform.
*   **4.2. Design `IntegratedStrategyManager`:**
    *   **Purpose:** Central hub ("Decision Agent") inheriting `StrategyManagerABC`.
    *   **Responsibilities:** Manage strategy instances, fetch data, call strategy `generate_signals`, implement *selectable* combination algorithms (Phase 3), apply final thresholding (-> `Dict[str, int]` for Orchestrator).
    *   **Configuration:** Needs settings for strategy instances, combination method, weights, thresholds, rules, etc.

## 5. Phase 3 (Post-Refactoring): Develop Multiple Signal Combination Methods

**Goal:** Implement several algorithms within `IntegratedStrategyManager` for flexibility.

*   **5.1. Method: Weighted Averaging with Confidence:** Combine signals based on static agent weights and dynamic confidence scores.
*   **5.2. Method: Dynamic Contextual Weighting:** Adjust agent weights based on market regime and/or recent agent performance before weighted averaging.
*   **5.3. Method: Rule-Based Priority System:** Apply configurable rules to prioritize or override signals based on specific conditions.
*   **5.4. (Optional Advanced) Method: Meta-Learner Ensemble:** Use a trained model to predict the final signal based on all agent outputs as features.

## 6. Phase 4 (Post-Refactoring): Implementation & Integration

**Goal:** Code the designed framework.

*   Implement standardized agent output format.
*   Implement `IntegratedStrategyManager` class and combination methods.
*   Implement configuration loading.
*   Update Orchestrator/Backtester to use the new manager.
*   Add unit and integration tests.
*   Add logging.
