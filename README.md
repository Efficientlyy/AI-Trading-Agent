# AI Trading Agent

[![Code Coverage](<coverage-badge-url>)](<coverage-report-url>)
[![Build Status](<build-status-badge-url>)](<build-status-url>)

This repository contains an AI Trading Agent designed for backtesting and potentially live trading financial strategies. The agent leverages data acquisition, feature engineering, and machine learning models to inform trading decisions.

## Project Status

Currently under active development. Key components include:

*   **Backend**:
    * **Data Acquisition**: Using `ccxt` to fetch historical market data.
    * **Data Processing**: Includes feature engineering and indicator calculations.
    * **Trading Engine**: Core components for managing orders, positions, and portfolio state.
    * **Backtesting Framework**: Comprehensive backtesting system with multi-asset support.
    * **Testing**: Extensive unit tests using `pytest` to ensure reliability.

*   **Frontend** (React/TypeScript):
    * **Authentication System**: Secure JWT-based authentication
    * **Dashboard**: Real-time portfolio and performance monitoring
    * **WebSocket Integration**: Live data updates with mock data support for development
    * **Responsive Design**: Works across desktop, tablet, and mobile devices
    * **Dark/Light Mode**: Theme customization

See `docs/PLAN.md` for the development roadmap and `docs/architecture.md` for a high-level overview.

## Key Features

*   Modular architecture
*   Pydantic models for data validation
*   Multi-asset backtesting with portfolio-level analysis
*   Performance metrics calculation (Sharpe ratio, Sortino ratio, drawdowns, etc.)
*   Rust acceleration for performance-critical components:
    * Technical indicators (SMA, EMA, MACD, RSI)
    * Lag features for time series analysis
    * Backtesting core loop
*   Modern React/TypeScript frontend with Tailwind CSS
*   Real-time data visualization and trading interface
*   Comprehensive test suite
*   Support for multiple data providers (planned)
*   Integration with various exchanges via `ccxt` (planned)

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

See `docs/usage_guide.md` for more details.

## Contributing

Contributions are welcome! Please refer to `docs/contributing.md` for guidelines on how to contribute to the project, including setting up a development environment and submitting pull requests.
