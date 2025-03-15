# Configuration Management

This document describes how to configure the AI Trading Agent system.

## Overview

The AI Trading Agent uses a hierarchical configuration system with support for:

- YAML configuration files
- Environment variable overrides
- Schema validation

This approach provides flexibility while ensuring configuration correctness.

## Configuration Files

The system uses YAML files for configuration. By default, it loads the `config/system.yaml` file at startup. You can also load additional configuration files at runtime.

Example configuration file:

```yaml
system:
  logging:
    level: INFO
    file_enabled: true
    file_path: logs/trading.log

exchanges:
  binance:
    enabled: true
    api_key: your_api_key
    api_secret: your_api_secret
    testnet: true

strategies:
  ma_crossover:
    enabled: true
    symbols:
      - BTC/USDT
      - ETH/USDT
    fast_period: 10
    slow_period: 30
```

## Environment Variables

You can override any configuration value using environment variables. The system follows this pattern:

```
COMPONENT_SECTION_KEY=value
```

For example, to override the Binance API key:

```
EXCHANGES_BINANCE_API_KEY=your_new_api_key
```

Environment variables can override nested configurations using underscore separators. The system will automatically convert values to the appropriate type based on the existing configuration or infer the type if the key doesn't exist.

## Schema Validation

The system supports schema validation to ensure your configuration is correct. We use a schema system that validates:

- Required fields
- Value types (string, integer, float, boolean, array, object)
- Value constraints (min/max for numeric values)
- Allowed values (for enum-like fields)
- Nested objects and arrays

### Creating a Schema

You can create a schema using the `ConfigSchema` and `ConfigValueSchema` classes:

```python
from src.common.config_schema import ConfigSchema, ConfigValueSchema, ConfigValueType

schema = ConfigSchema(
    title="Example Schema",
    description="An example schema",
    version="1.0.0",
    properties={
        "system": ConfigValueSchema(
            type=ConfigValueType.OBJECT,
            description="System settings",
            properties={
                "debug": ConfigValueSchema(
                    type=ConfigValueType.BOOLEAN,
                    description="Debug mode",
                    default=False
                )
            }
        )
    }
)
```

### Applying a Schema

To apply a schema to the configuration system:

```python
from src.common.config import config

# Create your schema
schema = create_my_schema()

# Set the schema for validation
config.set_schema(schema)

# Now configuration loading will validate against this schema
```

### Validation Results

When you validate a configuration, you get a `ValidationResult` object with:

- `is_valid`: Boolean indicating if the configuration is valid
- `errors`: List of validation error messages
- `config`: The validated configuration (if valid)

## Using the Configuration System

The configuration system is available through the singleton `config` instance:

```python
from src.common.config import config

# Get a configuration value (with a default fallback)
log_level = config.get("system.logging.level", "INFO")

# Set a configuration value
config.set("system.debug", True)

# Load a configuration file
config.load_config_file("config/custom.yaml")

# Get the entire configuration
full_config = config.get_all()
```

## Best Practices

1. **Use the schema validation**: Always define a schema for your components to catch configuration errors early.

2. **Provide sensible defaults**: Make your application work with minimal configuration by providing good defaults.

3. **Use environment variables for secrets**: Don't commit sensitive information like API keys to configuration files; use environment variables instead.

4. **Document your configuration**: Make sure all configuration options are well-documented with descriptions and examples.

5. **Validate early**: Validate your configuration during startup to fail fast if there are configuration errors.
