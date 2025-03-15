# Common Utilities

This directory contains common utilities used throughout the trading system.

## Configuration System

The `config.py` module provides a flexible configuration system that supports:

1. Loading configuration from YAML files
2. Overriding configuration from environment variables
3. Programmatic configuration updates

See `docs/DEVELOPMENT_SETUP.md` for usage guidelines.

## Logging System

The `logging.py` module sets up structured logging using structlog. It configures:

1. Console logging (human-readable in development, JSON in production)
2. File logging (optional, based on configuration)
3. Log levels (configurable per component)

## Datetime Utilities

The `datetime_utils.py` module provides modern timezone-aware datetime handling:

- `utc_now()`: Get current UTC time (replacement for deprecated `datetime.utcnow()`)
- `format_iso()`: Format datetime as ISO 8601 string
- `parse_iso()`: Parse ISO 8601 string to datetime
- `days_between()`: Calculate days between two datetimes

Example usage:

```python
from src.common.datetime_utils import utc_now, format_iso, parse_iso, days_between

# Get current UTC time
now = utc_now()

# Format for storage or transmission
iso_string = format_iso(now)  # "2025-03-03T21:35:41+00:00"

# Parse from string
dt = parse_iso("2025-03-01T00:00:00+00:00")

# Calculate days difference
days = days_between(dt, now)  # 2
```

Always use these utilities instead of directly using `datetime` functions to ensure consistent timezone handling across the application.
