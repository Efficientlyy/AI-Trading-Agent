# AI Crypto Trading Agent

This project implements an AI-powered crypto trading system with modules for market data collection, sentiment analysis, strategy execution, risk management, and portfolio optimization.

## Features

- **Data Collection**: Integration with multiple exchange APIs
- **Sentiment Analysis**: Process news, social media, and on-chain metrics
- **Strategy Framework**: Extensible strategy system with multiple implementations
- **Decision Engine**: Combines signals from technical and sentiment analysis
- **Execution Layer**: Optimized order execution with smart routing
- **Risk Management**: Comprehensive risk controls and position sizing
- **Dashboard**: Real-time monitoring and visualization

## New Features

### Modular Dashboard Architecture

We've refactored the dashboard to follow a modular architecture with improved maintainability:

- **Component-Based Structure**: Separation of concerns with dedicated modules for distinct functionality
- **Single Responsibility Principle**: Each file handles only one aspect of the system
- **Reduced File Sizes**: All modules kept under 300 lines for better readability and maintenance
- **Mock Data Generation**: Dedicated mock data generator for development and testing
- **Data Service Layer**: Abstraction for seamless switching between mock and real data
- **Authentication Module**: Separate module for secure user authentication and authorization
- **Runner Script**: Improved launcher with auto-detection of available ports

To run the new modular dashboard:

```bash
python run_modular_dashboard.py
```

The modular architecture includes:
- `src/dashboard/utils/auth.py`: Authentication and user management
- `src/dashboard/utils/data_service.py`: Data service abstraction layer
- `src/dashboard/utils/enums.py`: Centralized enums for the dashboard
- `src/dashboard/utils/mock_data.py`: Mock data generation for development
- `src/dashboard/modern_dashboard_refactored.py`: Main dashboard application

### API Key Management

We've added a comprehensive API key management system to the dashboard:

- **Secure Storage**: Encrypted storage for exchange API keys
- **Validation**: Real-time validation against actual exchange APIs
- **User Interface**: Intuitive panel for adding, viewing, and managing keys
- **Pagination**: Efficient handling of large numbers of API keys
- **Role-Based Access**: Admin-only access to sensitive credentials

#### API Key Validation

The system validates API keys against actual exchange APIs using the following methods:

- **Binance**: Uses account endpoint with HMAC-SHA256 authentication
- **Coinbase**: Uses accounts endpoint with timestamp-based signatures
- **Kraken**: Uses balance endpoint with incremental nonce verification
- **Other Exchanges**: Exchange-specific validation methods

#### Pagination

The API key list supports pagination to handle large numbers of keys:

- **Server-side pagination**: Efficient retrieval of only necessary data
- **Customizable page size**: Adjust items per page
- **Navigation controls**: Intuitive page navigation UI
- **Status indicators**: Shows current page, total items, and page count

## Automated Tests

The system includes comprehensive tests for the API key management functionality:

- **Validation Tests**: Verify that API key validation works correctly for each exchange
- **Pagination Tests**: Ensure the pagination system handles all edge cases
- **Error Handling Tests**: Confirm proper handling of invalid keys and API errors
- **Mock Response Tests**: Test behavior when exchanges are unavailable

## Setup and Installation

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Set up configuration in the `config/` directory
4. Run the system: `python run_trading_system.py`

## Dashboard

To run the dashboard:

```bash
python run_dashboard.py
```

The dashboard is available at `http://localhost:8050/` by default.

## Configuration

Configuration files are stored in the `config/` directory in YAML format.

## Development

### Developer Documentation

We've created comprehensive documentation to help new developers get up to speed quickly:

- **[Smart Path for New Developers](docs/SMART_PATH_FOR_NEW_DEVELOPERS.md)**: An optimized guide for understanding the project structure and working efficiently with the codebase
- **[Architecture Overview](ARCHITECTURE.md)**: Detailed system architecture and component relationships
- **[Dashboard Architecture](docs/DASHBOARD_ARCHITECTURE.md)**: Specific documentation for the dashboard system
- **[Dashboard Implementation](docs/DASHBOARD_IMPLEMENTATION.md)**: Technical details of the dashboard implementation
- **[Modern Dashboard Guide](docs/MODERN_DASHBOARD_GUIDE.md)**: User guide for the modern dashboard

### Requirements

- Python 3.9+
- Required packages in `requirements.txt`
- Rust compiler (for optimized components)

### Testing

Run tests with pytest:

```bash
python -m pytest
```

### Adding New Exchanges

To add support for a new exchange:

1. Create a connector in `src/data_collection/connectors/`
2. Implement the exchange interface in `src/execution/exchange/`
3. Add validation logic in `src/dashboard/modern_components.py`
4. Update the UI to include the new exchange option

## License

[License details here]
