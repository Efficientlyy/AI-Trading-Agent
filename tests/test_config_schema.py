"""Tests for configuration schema validation."""

import unittest
from typing import Dict, List, Optional

from src.common.config_schema import (
    ConfigSchema, ConfigValueSchema, ConfigValueType, create_example_schema
)


class TestConfigSchema(unittest.TestCase):
    """Tests for configuration schema validation."""
    
    def test_config_value_type_from_python_type(self):
        """Test conversion from Python types to ConfigValueType."""
        self.assertEqual(ConfigValueType.from_python_type(str), ConfigValueType.STRING)
        self.assertEqual(ConfigValueType.from_python_type(int), ConfigValueType.INTEGER)
        self.assertEqual(ConfigValueType.from_python_type(float), ConfigValueType.FLOAT)
        self.assertEqual(ConfigValueType.from_python_type(bool), ConfigValueType.BOOLEAN)
        self.assertEqual(ConfigValueType.from_python_type(dict), ConfigValueType.OBJECT)
        self.assertEqual(ConfigValueType.from_python_type(list), ConfigValueType.ARRAY)
        
        # Test unsupported type
        with self.assertRaises(ValueError):
            ConfigValueType.from_python_type(set)
    
    def test_config_value_schema_validation(self):
        """Test validation of ConfigValueSchema configuration."""
        # Test valid schema
        valid_schema = ConfigValueSchema(
            type=ConfigValueType.INTEGER,
            description="Test integer",
            min_value=0,
            max_value=100
        )
        self.assertEqual(valid_schema.type, ConfigValueType.INTEGER)
        self.assertEqual(valid_schema.min_value, 0)
        self.assertEqual(valid_schema.max_value, 100)
        
        # Test invalid min/max value with non-numeric type
        with self.assertRaises(ValueError):
            ConfigValueSchema(
                type=ConfigValueType.STRING,
                description="Test string",
                min_value=0  # Invalid for string type
            )
        
        # Test invalid properties with non-object type
        with self.assertRaises(ValueError):
            ConfigValueSchema(
                type=ConfigValueType.STRING,
                description="Test string",
                properties={"key": ConfigValueSchema(type=ConfigValueType.STRING, description="Invalid")}
            )
        
        # Test invalid items with non-array type
        with self.assertRaises(ValueError):
            ConfigValueSchema(
                type=ConfigValueType.STRING,
                description="Test string",
                items=ConfigValueSchema(type=ConfigValueType.STRING, description="Invalid")
            )
        
        # Test invalid allowed_values with object type
        with self.assertRaises(ValueError):
            ConfigValueSchema(
                type=ConfigValueType.OBJECT,
                description="Test object",
                allowed_values=[{"key": "value"}]  # Invalid for object type
            )
    
    def test_config_schema_validation(self):
        """Test validation of configuration against schema."""
        # Create a schema for testing
        schema = ConfigSchema(
            title="Test Schema",
            description="A schema for testing",
            version="1.0.0",
            properties={
                "string_value": ConfigValueSchema(
                    type=ConfigValueType.STRING,
                    description="A string value",
                    required=True
                ),
                "int_value": ConfigValueSchema(
                    type=ConfigValueType.INTEGER,
                    description="An integer value",
                    min_value=0,
                    max_value=100,
                    default=50
                ),
                "bool_value": ConfigValueSchema(
                    type=ConfigValueType.BOOLEAN,
                    description="A boolean value",
                    default=False
                ),
                "enum_value": ConfigValueSchema(
                    type=ConfigValueType.STRING,
                    description="An enum value",
                    allowed_values=["option1", "option2", "option3"]
                ),
                "nested_object": ConfigValueSchema(
                    type=ConfigValueType.OBJECT,
                    description="A nested object",
                    properties={
                        "nested_string": ConfigValueSchema(
                            type=ConfigValueType.STRING,
                            description="A nested string",
                            required=True
                        ),
                        "nested_int": ConfigValueSchema(
                            type=ConfigValueType.INTEGER,
                            description="A nested integer",
                            min_value=0
                        )
                    }
                ),
                "array_value": ConfigValueSchema(
                    type=ConfigValueType.ARRAY,
                    description="An array value",
                    items=ConfigValueSchema(
                        type=ConfigValueType.STRING,
                        description="Array item"
                    )
                )
            }
        )
        
        # Test valid configuration
        valid_config = {
            "string_value": "test",
            "int_value": 50,
            "bool_value": True,
            "enum_value": "option1",
            "nested_object": {
                "nested_string": "nested test",
                "nested_int": 10
            },
            "array_value": ["item1", "item2", "item3"]
        }
        errors = schema.validate_config(valid_config)
        self.assertEqual(errors, [])
        
        # Test missing required property
        missing_required = {
            "int_value": 50
            # Missing string_value which is required
        }
        errors = schema.validate_config(missing_required)
        self.assertEqual(len(errors), 1)
        self.assertIn("Missing required property: string_value", errors)
        
        # Test invalid type
        invalid_type = {
            "string_value": "test",
            "int_value": "not an integer"  # Should be an integer
        }
        errors = schema.validate_config(invalid_type)
        self.assertEqual(len(errors), 1)
        self.assertIn("int_value: Expected integer", errors[0])
        
        # Test value out of range
        out_of_range = {
            "string_value": "test",
            "int_value": 150  # Above max_value of 100
        }
        errors = schema.validate_config(out_of_range)
        self.assertEqual(len(errors), 1)
        self.assertIn("int_value: Value 150 is greater than maximum 100", errors[0])
        
        # Test invalid enum value
        invalid_enum = {
            "string_value": "test",
            "enum_value": "invalid_option"  # Not in allowed_values
        }
        errors = schema.validate_config(invalid_enum)
        self.assertEqual(len(errors), 1)
        self.assertIn("enum_value: Value invalid_option is not in allowed values", errors[0])
        
        # Test nested validation
        invalid_nested = {
            "string_value": "test",
            "nested_object": {
                # Missing nested_string which is required
                "nested_int": -10  # Below min_value of 0
            }
        }
        errors = schema.validate_config(invalid_nested)
        self.assertEqual(len(errors), 2)
        self.assertIn("nested_object.nested_string: Missing required property", errors)
        self.assertIn("nested_object.nested_int: Value -10 is less than minimum 0", errors)
        
        # Test array validation
        invalid_array = {
            "string_value": "test",
            "array_value": ["item1", 2, "item3"]  # Item at index 1 should be a string
        }
        errors = schema.validate_config(invalid_array)
        self.assertEqual(len(errors), 1)
        self.assertIn("array_value[1]: Expected string", errors[0])
    
    def test_example_schema(self):
        """Test the example schema creation."""
        schema = create_example_schema()
        
        # Basic schema checks
        self.assertEqual(schema.title, "Trading System Configuration")
        self.assertEqual(schema.version, "1.0.0")
        self.assertIn("system", schema.properties)
        self.assertIn("exchanges", schema.properties)
        self.assertIn("strategies", schema.properties)
        
        # Test a valid configuration
        valid_config = {
            "exchanges": {
                "binance": {
                    "enabled": True,
                    "api_key": "test_key",
                    "api_secret": "test_secret",
                    "testnet": True
                }
            },
            "strategies": {
                "ma_crossover": {
                    "enabled": True,
                    "symbols": ["BTC/USDT", "ETH/USDT"],
                    "fast_period": 15,
                    "slow_period": 50
                }
            }
        }
        errors = schema.validate_config(valid_config)
        self.assertEqual(errors, [])
        
        # Test an invalid configuration
        invalid_config = {
            "exchanges": {
                "binance": {
                    "enabled": True,
                    # Missing required api_key and api_secret
                    "testnet": "not a boolean"  # Should be a boolean
                }
            },
            "strategies": {
                "ma_crossover": {
                    "enabled": True,
                    "symbols": "BTC/USDT",  # Should be an array
                    "fast_period": 1,  # Below min_value
                    "slow_period": 600  # Above max_value
                }
            }
        }
        errors = schema.validate_config(invalid_config)
        self.assertGreater(len(errors), 0)
        self.assertIn("exchanges.binance.api_key: Missing required property", str(errors))
        self.assertIn("exchanges.binance.api_secret: Missing required property", str(errors))
        self.assertIn("exchanges.binance.testnet: Expected boolean", str(errors))
        self.assertIn("strategies.ma_crossover.symbols: Expected array", str(errors))
        self.assertIn("strategies.ma_crossover.fast_period: Value 1 is less than minimum 2", str(errors))
        self.assertIn("strategies.ma_crossover.slow_period: Value 600 is greater than maximum 500", str(errors))


if __name__ == "__main__":
    unittest.main()
