# AI Trading Agent

[![Code Coverage](<coverage-badge-url>)](<coverage-report-url>)
[![Build Status](<build-status-badge-url>)](<build-status-url>)

This repository contains an AI Trading Agent designed for backtesting and live trading financial strategies. The agent leverages advanced technical analysis, market regime detection, machine learning signal validation, and sentiment analysis to inform trading decisions.

## Project Status

Currently under active development. Key components include:

*   **Backend**:
    * **Data Acquisition**: Using multiple providers with failover mechanisms for robust data acquisition.
    * **Advanced Technical Analysis**: Enhanced technical analysis with pattern recognition, interactive charting, and regime-specific strategies with ML-enhanced signal validation.
    * **Market Regime Classification**: Automatic detection of trending, ranging, volatile, and transitional markets.
    * **Mock/Real Data Toggle**: Seamless switching between synthetic and real market data for testing and production, with integrated UI components and comprehensive API support.
    * **Trading Engine**: Advanced position management with dynamic scaling and risk adjustment.
    * **Backtesting Framework**: Comprehensive system with multi-asset support and portfolio-level analysis.
    * **Sentiment Analysis**: Integration of news sentiment data for enhanced trading signals.
    * **Testing**: Extensive unit and integration tests to ensure reliability.

*   **Frontend** (React/TypeScript):
    * **Authentication System**: Secure JWT-based authentication
    * **Dashboard**: Real-time portfolio and performance monitoring
    * **Trading Interface**: Enhanced order entry form with position information and interactive technical analysis tools
    * **Asset Visualization**: Interactive charts for portfolio allocation and equity curve
    * **Sentiment Analysis**: Comprehensive sentiment dashboard with signal distribution
    * **WebSocket Integration**: Live data updates with mock data support for development
    * **Responsive Design**: Works across desktop, tablet, and mobile devices
    * **Dark/Light Mode**: Theme customization
    * **Notification System**: Toast notifications for user actions and system alerts

See `docs/PLAN.md` for the development roadmap and `docs/architecture.md` for a high-level overview.

## Key Features

*   Modular architecture with a flexible agent-based design
*   Specialized strategies for different market regimes:
    * Trending market strategies focused on momentum
    * Range-bound strategies optimized for oscillating markets
    * Volatility breakout strategies for dynamic markets
    * Mean-reversion strategies for overextended markets
    * Regime transition strategies for changing market conditions
*   ML-enhanced signal validation with:
    * Ensemble model architecture for different market conditions
    * Advanced feature engineering with pattern recognition
    * Market regime adaptation for dynamic threshold adjustments
    * Confidence scoring with uncertainty quantification
*   Mock/Real data toggle for seamless switching between testing and production
*   Multi-asset backtesting with portfolio-level analysis and correlation-aware risk management
*   Performance metrics calculation (Sharpe ratio, Sortino ratio, drawdowns, etc.)
*   Rust acceleration for performance-critical components:
    * Technical indicators (SMA, EMA, MACD, RSI)
    * Lag features for time series analysis
    * Backtesting core loop
*   Health monitoring system with automated recovery mechanisms
*   Modern React/TypeScript frontend with Tailwind CSS
*   Real-time data visualization and trading interface
*   Comprehensive test suite with integration testing
*   Support for multiple data providers with failover mechanisms

## Project Structure

```
AI-Trading-Agent/
├── backend/             # Python backend for trading engine and backtesting
├── frontend/            # React/TypeScript frontend for dashboard and UI
├── docs/                # Documentation files
└── tests/               # Test suite
```

## Setup

### Backend

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd AI-Trading-Agent
    ```

2.  **Create a virtual environment:**
    ```bash
    python -m venv .venv
    # Activate the environment (Windows)
    .venv\Scripts\activate
    # Or (Linux/macOS)
    # source .venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up environment variables:**
    Copy `.env.example` to `.env` and fill in any necessary API keys or settings.

### Frontend

1.  **Navigate to the frontend directory:**
    ```bash
    cd frontend
    ```

2.  **Install dependencies:**
    ```bash
    npm install
    ```

3.  **Start the development server:**
    ```bash
    npm start
    ```

4.  **Open in browser:**
    Open [http://localhost:3000](http://localhost:3000) to view the application.

## Usage

### Running Tests

To run the full test suite:
```bash
pytest
```

To run tests for a specific module:
```bash
pytest tests/unit/trading_engine/
```

For verbose output:
```bash
pytest -v
```

### Running Backtests

To run a multi-asset backtest example:
```bash
python examples/multi_asset_backtest.py
```

This will run a moving average crossover strategy on multiple assets and generate performance metrics and visualizations.

### Technical Analysis and Mock/Real Data Toggle

#### Technical Analysis Dashboard

The Technical Analysis dashboard provides comprehensive market analysis tools:

- **Chart Analysis**: Interactive charting with multiple timeframes and indicators
- **Pattern Recognition**: Automatic detection of chart patterns with confidence scoring
- **Indicator Overlays**: Customizable technical indicators with parameter adjustment
- **Mock/Real Data Awareness**: Visual indicators when using mock data for testing

To access the Technical Analysis dashboard:

```bash
python -m ai_trading_agent.run_dashboard
```

Navigate to the "Technical Analysis" tab in the dashboard interface.

#### Testing the Mock/Real Data Toggle

To demonstrate the mock/real data toggle functionality:

```bash
python examples/mock_real_toggle_demo.py
```

This example shows how to switch between mock and real data sources and configure mock data parameters.

The toggle can be controlled via:

- **UI Toggle**: Use the toggle switch in the dashboard header
- **API Endpoints**: `/api/data-source/toggle` (POST) and `/api/data-source/status` (GET)
- **Programmatic Interface**: `DataSourceConfig` and `DataSourceFactory` classes

### Exploring Regime-Specific Strategies

To see how different market regime strategies work:
```bash
python examples/regime_strategies_demo.py
```

See `docs/usage_guide.md` for more details.

## Contributing

Contributions are welcome! Please refer to `docs/contributing.md` for guidelines on how to contribute to the project, including setting up a development environment and submitting pull requests.
