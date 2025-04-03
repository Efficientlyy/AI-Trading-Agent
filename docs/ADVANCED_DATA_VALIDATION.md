# Advanced Data Validation

This document provides an overview of the Advanced Data Validation system implemented in Phase 3 of the Real Data Integration project.

## Overview

The Advanced Data Validation system ensures data quality, detects anomalies, and prevents corrupt data from affecting analysis. It provides comprehensive validation capabilities beyond basic type checking, including range validation, temporal validation, cross-field validation, and anomaly detection.

## Architecture

The Advanced Data Validation system consists of the following components:

1. **Data Validator**: A Python class that performs validation against schemas and rules.
2. **Validation Rules**: Configurable rules that define validation criteria.
3. **Validation Schemas**: JSON Schema definitions for data structure validation.
4. **Validation UI**: User interface components for configuring and monitoring validation.

## Features

### Range Validation

- **Min/Max Value Checking**: Validates that numeric values fall within specified ranges.
- **Percentage Change Validation**: Detects extreme fluctuations in values.
- **Historical Trend Validation**: Compares values against historical trends.
- **Seasonal Pattern Recognition**: Validates values against seasonal patterns.

### Temporal Validation

- **Timestamp Sequence Validation**: Ensures timestamps are in chronological order.
- **Time Gap Detection**: Identifies missing data points in time series.
- **Timezone Handling**: Properly handles data from different timezones.
- **Date/Time Format Standardization**: Ensures consistent date/time formats.

### Cross-Field Validation

- **Relationship Validation**: Validates relationships between fields.
- **Logical Constraint Checking**: Ensures logical constraints are met.
- **Conditional Validation Rules**: Applies validation rules conditionally.
- **Complex Formula Validation**: Validates values against complex formulas.

### Anomaly Detection

- **Statistical Outlier Detection**: Identifies values that deviate significantly from the norm.
- **Pattern Deviation Recognition**: Detects deviations from established patterns.
- **Historical Comparison**: Compares values against historical data.
- **Machine Learning-Based Detection**: Uses ML models to identify anomalies.

## Implementation Details

### Data Validator

The `DataValidator` class is the core component of the validation system. It provides methods for validating data against schemas and rules, and for detecting anomalies.

```python
class DataValidator:
    def __init__(self, schema=None, validation_level=ValidationLevel.STANDARD):
        self.schema = schema
        self.validation_level = validation_level
        self.schema_validator = None
        
        # Initialize schema validator if schema is provided
        if self.schema:
            self.schema_validator = Draft7Validator(self.schema)
        
        # Historical data for anomaly detection
        self.historical_data = {}
        
        # Validation rules
        self.validation_rules = []
    
    def validate(self, data, context=None):
        # Initialize validation result
        result = ValidationResult(is_valid=True)
        
        # Initialize context
        ctx = context or {}
        
        try:
            # Perform schema validation if schema is available
            if self.schema_validator:
                self._validate_schema(data, result)
            
            # Perform basic validation
            self._validate_basic(data, result, ctx)
            
            # Perform standard validation if level is STANDARD or higher
            if self.validation_level in [ValidationLevel.STANDARD, ValidationLevel.STRICT]:
                self._validate_standard(data, result, ctx)
            
            # Perform strict validation if level is STRICT
            if self.validation_level == ValidationLevel.STRICT:
                self._validate_strict(data, result, ctx)
            
            # Apply custom validation rules
            self._apply_validation_rules(data, result, ctx)
            
            return result
        except Exception as e:
            result.add_error(f"Validation error: {str(e)}", severity=ValidationSeverity.CRITICAL)
            return result
```

### Validation Result

The `ValidationResult` class represents the result of a validation operation. It includes information about whether the data is valid, and any errors or warnings that were generated.

```python
class ValidationResult:
    def __init__(self, is_valid=True, errors=None, warnings=None):
        self.is_valid = is_valid
        self.errors = errors or []
        self.warnings = warnings or []
        self.timestamp = datetime.now().isoformat()
    
    def add_error(self, message, path=None, severity=ValidationSeverity.ERROR, details=None):
        self.errors.append({
            'message': message,
            'path': path,
            'severity': severity.name,
            'details': details or {},
            'timestamp': datetime.now().isoformat()
        })
        
        # If severity is ERROR or CRITICAL, mark as invalid
        if severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL]:
            self.is_valid = False
    
    def add_warning(self, message, path=None, details=None):
        self.warnings.append({
            'message': message,
            'path': path,
            'severity': ValidationSeverity.WARNING.name,
            'details': details or {},
            'timestamp': datetime.now().isoformat()
        })
```

### Validation Levels

The validation system supports three levels of validation:

1. **BASIC**: Performs basic type and format validation.
2. **STANDARD**: Adds range validation, temporal validation, and cross-field validation.
3. **STRICT**: Adds anomaly detection and more rigorous validation.

```python
class ValidationLevel(Enum):
    BASIC = 1
    STANDARD = 2
    STRICT = 3
```

### Validation Severity

Validation issues are categorized by severity:

1. **INFO**: Informational messages that don't affect validity.
2. **WARNING**: Potential issues that don't invalidate the data.
3. **ERROR**: Issues that invalidate the data.
4. **CRITICAL**: Severe issues that require immediate attention.

```python
class ValidationSeverity(Enum):
    INFO = 1
    WARNING = 2
    ERROR = 3
    CRITICAL = 4
```

## Usage

### Basic Validation

```python
# Create validator with schema
schema = {
    "type": "object",
    "required": ["symbol", "price", "timestamp"],
    "properties": {
        "symbol": {"type": "string"},
        "price": {"type": "number", "minimum": 0},
        "timestamp": {"type": "string", "format": "date-time"}
    }
}
validator = DataValidator(schema=schema)

# Validate data
data = {
    "symbol": "BTC/USD",
    "price": 45000,
    "timestamp": "2025-03-30T12:34:56Z"
}
result = validator.validate(data)

# Check result
if result.is_valid:
    print("Data is valid")
else:
    print("Data is invalid:")
    for error in result.errors:
        print(f"- {error['message']}")
```

### Advanced Validation

```python
# Create validator with schema and strict validation level
validator = DataValidator(schema=schema, validation_level=ValidationLevel.STRICT)

# Add historical data for anomaly detection
historical_prices = [42000, 43000, 44000, 43500, 44500]
validator.add_historical_data("price", historical_prices)

# Add custom validation rule
def validate_price_change(data, context):
    if "previous_price" in context and "price" in data:
        change_percent = abs(data["price"] - context["previous_price"]) / context["previous_price"] * 100
        if change_percent > 5:
            return {
                "is_valid": True,  # Warning doesn't invalidate data
                "message": f"Price change of {change_percent:.2f}% exceeds 5%",
                "severity": "WARNING",
                "details": {
                    "current_price": data["price"],
                    "previous_price": context["previous_price"],
                    "change_percent": change_percent
                }
            }
    return True

validator.add_validation_rule(validate_price_change, "price_change")

# Validate data with context
context = {
    "previous_price": 42000
}
result = validator.validate(data, context)

# Check result
if result.is_valid:
    print("Data is valid")
    if result.warnings:
        print("Warnings:")
        for warning in result.warnings:
            print(f"- {warning['message']}")
else:
    print("Data is invalid:")
    for error in result.errors:
        print(f"- {error['message']}")
```

## UI Components

The Advanced Data Validation system includes UI components for configuring and monitoring validation:

1. **Validation Status Indicator**: Shows the current validation status (valid, warning, invalid, pending).
2. **Validation Results Panel**: Displays validation results, including errors and warnings.
3. **Validation Rules Panel**: Allows enabling/disabling and configuring validation rules.
4. **Validation Schema Editor**: Provides an interface for editing validation schemas.
5. **Validation Level Selector**: Allows selecting the validation level.

## API Endpoints

The system provides the following API endpoints:

1. **GET /api/validation/results**: Get validation results for all data sources.
2. **GET /api/validation/results/{source}**: Get validation results for a specific data source.
3. **GET /api/validation/rules**: Get all validation rules.
4. **PUT /api/validation/rules/{rule_id}**: Update a validation rule.
5. **GET /api/validation/schemas**: Get all validation schemas.
6. **PUT /api/validation/schemas/{schema_id}**: Update a validation schema.
7. **PUT /api/validation/level**: Set the validation level.
8. **POST /api/validation/refresh/{source}**: Refresh validation for a specific data source.

## Integration with Other Components

The Advanced Data Validation system integrates with other components of the Real Data Integration project:

1. **Real-time Data Updates**: Validates data in real-time as it arrives.
2. **Data Transformation Pipeline**: Validates data before and after transformation.
3. **Admin Controls**: Provides administrative controls for validation configuration.

## Future Enhancements

Planned enhancements for the Advanced Data Validation system include:

1. **Machine Learning-Based Anomaly Detection**: Implement ML models for more sophisticated anomaly detection.
2. **Automated Rule Generation**: Generate validation rules based on historical data patterns.
3. **Validation Dashboards**: Create dashboards for monitoring validation metrics.
4. **Validation Alerts**: Send alerts when critical validation issues are detected.
5. **Validation Reporting**: Generate reports on validation results over time.

## Conclusion

The Advanced Data Validation system provides comprehensive validation capabilities that ensure data quality and integrity. By detecting anomalies and preventing corrupt data from affecting analysis, it enhances the reliability and trustworthiness of the AI Trading Agent dashboard.