# Fee Management System

The Fee Management System is a comprehensive solution for tracking, analyzing, and optimizing trading fees across multiple cryptocurrency exchanges. It provides tools to help traders and trading systems make cost-effective decisions about exchange selection and order placement strategies.

## Key Features

- **Fee Tracking**: Record and store trading fees from all transactions across exchanges
- **Fee Estimation**: Estimate fees for planned transactions before execution
- **Fee Scheduling**: Maintain up-to-date fee schedules for different exchanges
- **Volume-based Tiers**: Support for exchange fee tiers based on trading volume
- **Fee Discounts**: Track and apply exchange-specific fee discounts
- **Fee Analytics**: Generate summaries and reports on fee expenditures
- **Fee Optimization**: Recommend optimal exchange allocation for trading strategies

## Components

### Models (`models.py`)

Defines the core data structures:

- `FeeType`: Enum for different types of fees (maker, taker, withdrawal, etc.)
- `FeeCalculationType`: Enum for different fee calculation methods
- `FeeTier`: Volume-based fee tier with maker/taker rates
- `FeeSchedule`: Complete fee schedule for an exchange
- `FeeDiscount`: Fee discount applied to an account
- `TransactionFee`: Record of a fee paid for a transaction
- `FeeEstimate`: Estimated fee for a planned transaction
- `FeeSummary`: Summary of fees paid over a time period

### Service (`service.py`)

Implements the core functionality:

- Loading and saving fee data (schedules, discounts, transactions)
- Updating fee schedules and discounts
- Recording transaction fees
- Calculating fee rates based on volume tiers and discounts
- Estimating fees for planned transactions
- Generating fee summaries and reports
- Optimizing exchange allocation based on fees

### API (`api.py`)

Provides a simplified interface for other components:

- Getting and updating fee schedules
- Managing fee discounts
- Estimating transaction fees
- Recording fees
- Generating fee summaries
- Optimizing exchange allocation

## Usage Examples

### Initializing the Fee Management System

```python
from src.fees.api import FeeManagementAPI

# Initialize the API
fee_api = FeeManagementAPI()
```

### Setting Up Fee Schedules

```python
# Define a fee schedule for an exchange
binance_schedule = {
    "default_maker_fee": 0.001,  # 0.1%
    "default_taker_fee": 0.001,  # 0.1%
    "calculation_type": "percentage",
    "tiers": [
        {
            "min_volume": 0,
            "max_volume": 50000,
            "maker_fee": 0.001,
            "taker_fee": 0.001,
            "description": "Default tier"
        },
        {
            "min_volume": 50000,
            "max_volume": 500000,
            "maker_fee": 0.0008,
            "taker_fee": 0.001,
            "description": "VIP 1"
        }
    ],
    "withdrawal_fees": {
        "BTC": 0.0005,
        "ETH": 0.005,
        "USDT": 1.0
    }
}

# Update the fee schedule
fee_api.update_fee_schedule("binance", binance_schedule)
```

### Adding Fee Discounts

```python
# Define a fee discount
discount = {
    "exchange_id": "binance",
    "discount_percentage": 25.0,
    "applies_to": ["maker", "taker"],
    "reason": "BNB holding discount",
    "expiry": "2023-12-31T23:59:59"  # ISO format string
}

# Add the discount
fee_api.add_fee_discount(discount)
```

### Estimating Transaction Fees

```python
# Estimate fee for a transaction
fee_estimate = fee_api.estimate_transaction_fee(
    exchange_id="binance",
    fee_type="maker",
    asset="BTC",
    amount=1.0,
    price=40000,
    volume_30d=100000  # Optional 30-day volume for tier calculation
)

print(f"Estimated fee: {fee_estimate['estimated_amount']} {fee_estimate['asset']}")
print(f"USD value: ${fee_estimate['usd_value']}")
```

### Recording Transaction Fees

```python
# Record a fee for a completed transaction
fee_data = {
    "transaction_id": "ord12345",
    "exchange_id": "binance",
    "fee_type": "taker",
    "asset": "BTC",
    "amount": 0.001,
    "usd_value": 40.0,
    "transaction_time": "2023-05-01T12:34:56",
    "related_to": "btc_grid_strategy",
    "details": {
        "trade_amount": 1.0,
        "price": 40000
    }
}

fee_api.record_fee(fee_data)
```

### Generating Fee Summaries

```python
# Get a summary of fees paid over a time period
from datetime import datetime, timedelta

now = datetime.now()
start_time = now - timedelta(days=30)

summary = fee_api.get_fee_summary(
    start_time=start_time.isoformat(),
    end_time=now.isoformat(),
    exchange_ids=["binance", "coinbase"],  # Optional filter by exchange
    fee_types=["maker", "taker"],          # Optional filter by fee type
    assets=["BTC", "ETH"],                 # Optional filter by asset
    related_to="btc_grid_strategy"         # Optional filter by strategy
)

print(f"Total fees: ${summary['total_fees_usd']}")
print("Fees by exchange:", summary['by_exchange'])
print("Fees by type:", summary['by_type'])
print("Fees by asset:", summary['by_asset'])
```

### Optimizing Exchange Allocation

```python
# Define trading strategies with expected volumes
strategies = [
    {
        "id": "btc_grid_strategy",
        "asset_pair": "BTC/USDT",
        "monthly_volume": 500000,
        "description": "BTC Grid Trading Strategy"
    },
    {
        "id": "eth_momentum_strategy",
        "asset_pair": "ETH/USDT",
        "monthly_volume": 250000,
        "description": "ETH Momentum Strategy"
    }
]

# List of available exchanges
exchanges = ["binance", "coinbase", "kraken"]

# Constraints on exchange usage
constraints = {
    "max_exchanges_per_strategy": 1,
    "required_exchanges": {},
    "excluded_exchanges": {}
}

# Get the optimized allocation
allocation = fee_api.optimize_exchange_allocation(
    strategies=strategies,
    exchanges=exchanges,
    constraints=constraints
)

print("Optimized exchange allocation:")
for strategy_id, details in allocation['allocation'].items():
    print(f"  {strategy_id}:")
    print(f"    Best exchange: {details['exchange']}")
    print(f"    Estimated fee rate: {details['estimated_fee_rate']*100:.4f}%")
    print(f"    Estimated monthly fee: ${details['estimated_monthly_fee']:.2f}")

print(f"Total estimated monthly fees: ${allocation['estimated_total_monthly_fee']:.2f}")
```

## Integration

The Fee Management System can be integrated with:

1. **Order Management System**: To automatically record fees for executed orders
2. **Exchange Connectors**: To fetch and update fee schedules from exchanges
3. **Portfolio Management System**: To track fee expenses by strategy or portfolio
4. **Reporting System**: To include fee analysis in performance reports
5. **Strategy Execution**: To intelligently route orders based on fee optimization

## Future Enhancements

- Support for more complex fee models (e.g., tiered withdrawal fees)
- Integration with exchange APIs for real-time fee structure updates
- Machine learning-based fee prediction and optimization
- Support for tax calculations related to trading fees
- More sophisticated visualization and reporting tools 