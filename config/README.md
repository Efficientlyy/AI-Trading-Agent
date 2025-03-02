# Configuration System

The system uses YAML configuration files for each component, with environment variables for sensitive information.

## Configuration Files

- `system.yaml` - Global system settings
- `data_collection.yaml` - Data collection settings
- `analysis_agents.yaml` - Analysis agent configurations
- `decision_engine.yaml` - Decision engine settings
- `virtual_trading.yaml` - Virtual trading environment settings
- `notification.yaml` - Notification system settings
- `dashboard.yaml` - Dashboard configuration
- `market_regime.yaml` - Market regime classification settings
- `position_management.yaml` - Position management settings
- `portfolio_management.yaml` - Portfolio diversification settings
- `circuit_breakers.yaml` - Circuit breakers and drawdown protection
- `execution_quality.yaml` - Execution quality analysis settings
- `strategy_evolution.yaml` - Automated strategy evolution settings

## Configuration Structure

Each configuration file follows a similar structure:

```yaml
# Component name
component_name:
  # General settings
  general:
    setting1: value1
    setting2: value2
  
  # Subcomponent settings
  subcomponent1:
    setting1: value1
    setting2: value2
  
  # Feature toggles
  features:
    feature1: true
    feature2: false
```

## Environment Variables

Sensitive information (API keys, secrets, tokens) is stored in environment variables, not in configuration files. Use the `.env` file (copied from `.env.example`) to set these values.

## Configuration Validation

All configuration files are validated against schemas when the system starts. Invalid configuration will prevent startup.

## Overriding Configuration

Configuration can be overridden at runtime by setting environment variables using the pattern:
`COMPONENT_SUBCOMPONENT_SETTING=value`

For example, to override `data_collection.binance.max_requests_per_minute`, set:
`DATA_COLLECTION_BINANCE_MAX_REQUESTS_PER_MINUTE=100` 