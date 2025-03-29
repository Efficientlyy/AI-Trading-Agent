"""Configuration schema validation.

This module provides classes and utilities for validating configuration schemas.
It ensures that configuration values match expected types and constraints.
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Type, Union

from src.common.pydantic_compat import BaseModel, Field, ValidationInfo, field_validator


class ConfigValueType(str, Enum):
    """Types of configuration values."""
    
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    OBJECT = "object"
    ARRAY = "array"
    
    @classmethod
    def from_python_type(cls, python_type: Type) -> "ConfigValueType":
        """Convert a Python type to a ConfigValueType.
        
        Args:
            python_type: Python type to convert
            
        Returns:
            ConfigValueType corresponding to the Python type
            
        Raises:
            ValueError: If the Python type is not supported
        """
        if python_type == str:
            return cls.STRING
        elif python_type == int:
            return cls.INTEGER
        elif python_type == float:
            return cls.FLOAT
        elif python_type == bool:
            return cls.BOOLEAN
        elif python_type == dict:
            return cls.OBJECT
        elif python_type == list:
            return cls.ARRAY
        else:
            raise ValueError(f"Unsupported Python type: {python_type}")


class ConfigValueSchema(BaseModel):
    """Schema for a configuration value."""
    
    type: ConfigValueType
    description: str
    default: Optional[Any] = None
    required: bool = False
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    allowed_values: Optional[List[Any]] = None
    properties: Optional[Dict[str, "ConfigValueSchema"]] = None
    items: Optional["ConfigValueSchema"] = None
    
    @field_validator("min_value", "max_value")
    @classmethod
    def validate_numeric_constraints(cls, v, info: ValidationInfo):
        """Validate that min_value and max_value are only used with numeric types."""
        if v is not None and "type" in info.data and info.data["type"] not in (ConfigValueType.INTEGER, ConfigValueType.FLOAT):
            raise ValueError("min_value and max_value can only be used with numeric types")
        return v
    
    @field_validator("properties")
    @classmethod
    def validate_properties(cls, v, info: ValidationInfo):
        """Validate that properties is only used with object type."""
        if v is not None and "type" in info.data and info.data["type"] != ConfigValueType.OBJECT:
            raise ValueError("properties can only be used with object type")
        return v
    
    @field_validator("items")
    @classmethod
    def validate_items(cls, v, info: ValidationInfo):
        """Validate that items is only used with array type."""
        if v is not None and "type" in info.data and info.data["type"] != ConfigValueType.ARRAY:
            raise ValueError("items can only be used with array type")
        return v
    
    @field_validator("allowed_values")
    @classmethod
    def validate_allowed_values(cls, v, info: ValidationInfo):
        """Validate that allowed_values is only used with primitive types."""
        if v is not None and "type" in info.data and info.data["type"] in (ConfigValueType.OBJECT, ConfigValueType.ARRAY):
            raise ValueError("allowed_values cannot be used with object or array types")
        return v


class ConfigSchema(BaseModel):
    """Schema for a configuration."""
    
    title: str
    description: str
    version: str
    properties: Dict[str, ConfigValueSchema]
    
    def validate_config(self, config: Dict[str, Any]) -> List[str]:
        """Validate a configuration against this schema.
        
        Args:
            config: Configuration to validate
            
        Returns:
            List of validation errors, empty if valid
        """
        errors = []
        
        # Check for required properties
        for key, schema in self.properties.items():
            if schema.required and key not in config:
                errors.append(f"Missing required property: {key}")
        
        # Validate existing properties
        for key, value in config.items():
            if key in self.properties:
                property_errors = self._validate_value(value, self.properties[key], key)
                errors.extend(property_errors)
        
        return errors
    
    def _validate_value(self, value: Any, schema: ConfigValueSchema, path: str) -> List[str]:
        """Recursively validate a value against a schema.
        
        Args:
            value: Value to validate
            schema: Schema to validate against
            path: Path to the value in the configuration (for error messages)
            
        Returns:
            List of validation errors, empty if valid
        """
        errors = []
        
        # Check type
        if schema.type == ConfigValueType.STRING:
            if not isinstance(value, str):
                errors.append(f"{path}: Expected string, got {type(value).__name__}")
        elif schema.type == ConfigValueType.INTEGER:
            if not isinstance(value, int) or isinstance(value, bool):
                errors.append(f"{path}: Expected integer, got {type(value).__name__}")
            elif schema.min_value is not None and value < schema.min_value:
                errors.append(f"{path}: Value {value} is less than minimum {schema.min_value}")
            elif schema.max_value is not None and value > schema.max_value:
                errors.append(f"{path}: Value {value} is greater than maximum {schema.max_value}")
        elif schema.type == ConfigValueType.FLOAT:
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                errors.append(f"{path}: Expected float, got {type(value).__name__}")
            elif schema.min_value is not None and value < schema.min_value:
                errors.append(f"{path}: Value {value} is less than minimum {schema.min_value}")
            elif schema.max_value is not None and value > schema.max_value:
                errors.append(f"{path}: Value {value} is greater than maximum {schema.max_value}")
        elif schema.type == ConfigValueType.BOOLEAN:
            if not isinstance(value, bool):
                errors.append(f"{path}: Expected boolean, got {type(value).__name__}")
        elif schema.type == ConfigValueType.OBJECT:
            if not isinstance(value, dict):
                errors.append(f"{path}: Expected object, got {type(value).__name__}")
            elif schema.properties:
                for prop_key, prop_schema in schema.properties.items():
                    if prop_schema.required and prop_key not in value:
                        errors.append(f"{path}.{prop_key}: Missing required property")
                    elif prop_key in value:
                        prop_errors = self._validate_value(
                            value[prop_key], prop_schema, f"{path}.{prop_key}"
                        )
                        errors.extend(prop_errors)
        elif schema.type == ConfigValueType.ARRAY:
            if not isinstance(value, list):
                errors.append(f"{path}: Expected array, got {type(value).__name__}")
            elif schema.items:
                for i, item in enumerate(value):
                    item_errors = self._validate_value(
                        item, schema.items, f"{path}[{i}]"
                    )
                    errors.extend(item_errors)
        
        # Check allowed values
        if schema.allowed_values is not None and value not in schema.allowed_values:
            errors.append(
                f"{path}: Value {value} is not in allowed values: {schema.allowed_values}"
            )
        
        return errors


# Example usage
def create_example_schema() -> ConfigSchema:
    """Create an example configuration schema.
    
    Returns:
        Example schema for the trading system
    """
    return ConfigSchema(
        title="Trading System Configuration",
        description="Configuration schema for the AI Trading Agent system",
        version="1.0.0",
        properties={
            "system": ConfigValueSchema(
                type=ConfigValueType.OBJECT,
                description="System-wide settings",
                properties={
                    "logging": ConfigValueSchema(
                        type=ConfigValueType.OBJECT,
                        description="Logging settings",
                        properties={
                            "level": ConfigValueSchema(
                                type=ConfigValueType.STRING,
                                description="Logging level",
                                default="INFO",
                                allowed_values=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
                            ),
                            "file_enabled": ConfigValueSchema(
                                type=ConfigValueType.BOOLEAN,
                                description="Whether to log to a file",
                                default=False
                            ),
                            "file_path": ConfigValueSchema(
                                type=ConfigValueType.STRING,
                                description="Path to log file",
                                default="logs/trading.log"
                            )
                        }
                    )
                }
            ),
            "exchanges": ConfigValueSchema(
                type=ConfigValueType.OBJECT,
                description="Exchange configurations",
                properties={
                    "binance": ConfigValueSchema(
                        type=ConfigValueType.OBJECT,
                        description="Binance exchange settings",
                        properties={
                            "enabled": ConfigValueSchema(
                                type=ConfigValueType.BOOLEAN,
                                description="Whether Binance is enabled",
                                default=True,
                                required=True
                            ),
                            "api_key": ConfigValueSchema(
                                type=ConfigValueType.STRING,
                                description="Binance API key",
                                required=True
                            ),
                            "api_secret": ConfigValueSchema(
                                type=ConfigValueType.STRING,
                                description="Binance API secret",
                                required=True
                            ),
                            "testnet": ConfigValueSchema(
                                type=ConfigValueType.BOOLEAN,
                                description="Whether to use testnet",
                                default=True
                            )
                        }
                    )
                }
            ),
            "strategies": ConfigValueSchema(
                type=ConfigValueType.OBJECT,
                description="Strategy configurations",
                properties={
                    "ma_crossover": ConfigValueSchema(
                        type=ConfigValueType.OBJECT,
                        description="Moving average crossover strategy settings",
                        properties={
                            "enabled": ConfigValueSchema(
                                type=ConfigValueType.BOOLEAN,
                                description="Whether the strategy is enabled",
                                default=False
                            ),
                            "symbols": ConfigValueSchema(
                                type=ConfigValueType.ARRAY,
                                description="Symbols to trade",
                                items=ConfigValueSchema(
                                    type=ConfigValueType.STRING,
                                    description="Trading symbol"
                                ),
                                default=["BTC/USDT", "ETH/USDT"]
                            ),
                            "fast_period": ConfigValueSchema(
                                type=ConfigValueType.INTEGER,
                                description="Fast moving average period",
                                default=10,
                                min_value=2,
                                max_value=200
                            ),
                            "slow_period": ConfigValueSchema(
                                type=ConfigValueType.INTEGER,
                                description="Slow moving average period",
                                default=30,
                                min_value=5,
                                max_value=500
                            )
                        }
                    )
                }
            )
        }
    )
