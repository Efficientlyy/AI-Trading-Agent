# Market Regime Detection API

This API provides endpoints for detecting market regimes and backtesting trading strategies based on regime detection.

## Features

- Detect market regimes using various methods (volatility, momentum, HMM, etc.)
- Backtest trading strategies using detected regimes
- Visualize regime transitions and performance metrics
- Support for single and multi-asset analysis

## Installation

1. Install dependencies:
```bash
pip install fastapi uvicorn pandas numpy matplotlib seaborn
```

2. Make sure the market regime detection modules are in your Python path.

## Usage

### Starting the API server

```bash
cd src/api
python regime_detection_api.py
```

The API will be available at http://localhost:8000

### API Documentation

Once the server is running, you can access the interactive API documentation at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## API Endpoints

### GET /

Returns basic information about the API.

### GET /health

Health check endpoint.

### GET /methods

Returns available regime detection methods and trading strategies.

### POST /detect-regimes

Detects market regimes based on provided market data.

Example request:
```json
{
  "market_data": {
    "symbol": "SPY",
    "data": [
      {
        "date": "2023-01-01T00:00:00Z",
        "price": 380.5,
        "volume": 1000000,
        "return_value": 0.01
      },
      ...
    ]
  },
  "methods": ["volatility", "momentum", "hmm"],
  "lookback_window": 63,
  "include_statistics": true,
  "include_visualization": true
}
```

### POST /backtest

Runs a backtest using regime detection.

Example request:
```json
{
  "market_data": {
    "symbol": "SPY",
    "data": [
      {
        "date": "2023-01-01T00:00:00Z",
        "price": 380.5,
        "volume": 1000000,
        "return_value": 0.01
      },
      ...
    ]
  },
  "strategy_type": "momentum",
  "regime_methods": ["volatility", "momentum"],
  "train_test_split": 0.7,
  "walk_forward": true
}
```

## Error Handling

The API returns appropriate HTTP status codes and error messages for different types of errors:

- 400: Bad Request - Invalid input data
- 404: Not Found - Resource not found
- 500: Internal Server Error - Server-side error

## Development

To contribute to this API:

1. Fork the repository
2. Create a feature branch
3. Add your changes
4. Submit a pull request

## License

This project is licensed under the MIT License. 