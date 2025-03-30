"""
Data Validator

This module provides advanced data validation capabilities for the dashboard.
"""

import json
import logging
import re
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
from jsonschema import Draft7Validator, ValidationError, validators

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("data_validator")

class ValidationLevel(Enum):
    """Validation level enum"""
    BASIC = 1
    STANDARD = 2
    STRICT = 3

class ValidationSeverity(Enum):
    """Validation severity enum"""
    INFO = 1
    WARNING = 2
    ERROR = 3
    CRITICAL = 4

class ValidationResult:
    """Validation result class"""
    
    def __init__(self, 
                 is_valid: bool, 
                 errors: List[Dict[str, Any]] = None,
                 warnings: List[Dict[str, Any]] = None):
        """
        Initialize validation result.
        
        Args:
            is_valid: Whether the data is valid
            errors: List of validation errors
            warnings: List of validation warnings
        """
        self.is_valid = is_valid
        self.errors = errors or []
        self.warnings = warnings or []
        self.timestamp = datetime.now().isoformat()
    
    def add_error(self, 
                  message: str, 
                  path: str = None, 
                  severity: ValidationSeverity = ValidationSeverity.ERROR,
                  details: Dict[str, Any] = None):
        """
        Add an error to the validation result.
        
        Args:
            message: Error message
            path: JSON path to the error location
            severity: Error severity
            details: Additional error details
        """
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
    
    def add_warning(self, 
                    message: str, 
                    path: str = None, 
                    details: Dict[str, Any] = None):
        """
        Add a warning to the validation result.
        
        Args:
            message: Warning message
            path: JSON path to the warning location
            details: Additional warning details
        """
        self.warnings.append({
            'message': message,
            'path': path,
            'severity': ValidationSeverity.WARNING.name,
            'details': details or {},
            'timestamp': datetime.now().isoformat()
        })
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert validation result to dictionary.
        
        Returns:
            Dictionary representation of validation result
        """
        return {
            'is_valid': self.is_valid,
            'errors': self.errors,
            'warnings': self.warnings,
            'timestamp': self.timestamp
        }
    
    def __str__(self) -> str:
        """
        String representation of validation result.
        
        Returns:
            String representation
        """
        return json.dumps(self.to_dict(), indent=2)

class DataValidator:
    """
    Advanced data validator for ensuring data quality and integrity.
    """
    
    def __init__(self, 
                 schema: Dict[str, Any] = None, 
                 validation_level: ValidationLevel = ValidationLevel.STANDARD):
        """
        Initialize data validator.
        
        Args:
            schema: JSON schema for validation
            validation_level: Validation level
        """
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
        
        logger.info(f"Data validator initialized with level: {validation_level.name}")
    
    def set_schema(self, schema: Dict[str, Any]):
        """
        Set JSON schema for validation.
        
        Args:
            schema: JSON schema
        """
        self.schema = schema
        self.schema_validator = Draft7Validator(self.schema)
        logger.info("Schema validator updated")
    
    def add_validation_rule(self, rule_func: callable, name: str = None):
        """
        Add a custom validation rule.
        
        Args:
            rule_func: Validation rule function
            name: Rule name
        """
        rule_name = name or rule_func.__name__
        self.validation_rules.append({
            'name': rule_name,
            'func': rule_func
        })
        logger.info(f"Added validation rule: {rule_name}")
    
    def add_historical_data(self, key: str, data: Any):
        """
        Add historical data for anomaly detection.
        
        Args:
            key: Data key
            data: Historical data
        """
        self.historical_data[key] = data
        logger.info(f"Added historical data for key: {key}")
    
    def validate(self, data: Any, context: Dict[str, Any] = None) -> ValidationResult:
        """
        Validate data against schema and rules.
        
        Args:
            data: Data to validate
            context: Validation context
            
        Returns:
            Validation result
        """
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
            
            logger.info(f"Validation completed: valid={result.is_valid}, errors={len(result.errors)}, warnings={len(result.warnings)}")
            
            return result
        except Exception as e:
            logger.error(f"Error during validation: {e}")
            result.add_error(f"Validation error: {str(e)}", severity=ValidationSeverity.CRITICAL)
            return result
    
    def _validate_schema(self, data: Any, result: ValidationResult):
        """
        Validate data against JSON schema.
        
        Args:
            data: Data to validate
            result: Validation result
        """
        try:
            errors = list(self.schema_validator.iter_errors(data))
            
            for error in errors:
                path = '/'.join([str(p) for p in error.path]) if error.path else None
                result.add_error(
                    message=error.message,
                    path=path,
                    severity=ValidationSeverity.ERROR,
                    details={'schema_path': '/'.join([str(p) for p in error.schema_path])}
                )
        except Exception as e:
            logger.error(f"Schema validation error: {e}")
            result.add_error(f"Schema validation error: {str(e)}", severity=ValidationSeverity.ERROR)
    
    def _validate_basic(self, data: Any, result: ValidationResult, context: Dict[str, Any]):
        """
        Perform basic validation checks.
        
        Args:
            data: Data to validate
            result: Validation result
            context: Validation context
        """
        # Check if data is None
        if data is None:
            result.add_error("Data is None", severity=ValidationSeverity.ERROR)
            return
        
        # Check if data is empty
        if isinstance(data, (dict, list, str)) and len(data) == 0:
            result.add_warning("Data is empty")
        
        # Check data type based on context
        expected_type = context.get('expected_type')
        if expected_type:
            if not isinstance(data, expected_type):
                result.add_error(
                    f"Invalid data type: expected {expected_type.__name__}, got {type(data).__name__}",
                    severity=ValidationSeverity.ERROR
                )
    
    def _validate_standard(self, data: Any, result: ValidationResult, context: Dict[str, Any]):
        """
        Perform standard validation checks.
        
        Args:
            data: Data to validate
            result: Validation result
            context: Validation context
        """
        # Validate dictionary data
        if isinstance(data, dict):
            self._validate_dict(data, result, context)
        
        # Validate list data
        elif isinstance(data, list):
            self._validate_list(data, result, context)
        
        # Validate numeric data
        elif isinstance(data, (int, float)):
            self._validate_numeric(data, result, context)
        
        # Validate string data
        elif isinstance(data, str):
            self._validate_string(data, result, context)
        
        # Validate datetime data
        elif isinstance(data, datetime):
            self._validate_datetime(data, result, context)
        
        # Validate pandas DataFrame
        elif isinstance(data, pd.DataFrame):
            self._validate_dataframe(data, result, context)
    
    def _validate_strict(self, data: Any, result: ValidationResult, context: Dict[str, Any]):
        """
        Perform strict validation checks.
        
        Args:
            data: Data to validate
            result: Validation result
            context: Validation context
        """
        # Perform anomaly detection if historical data is available
        data_key = context.get('data_key')
        if data_key and data_key in self.historical_data:
            self._detect_anomalies(data, data_key, result, context)
        
        # Validate cross-field relationships
        if isinstance(data, dict) and 'relationships' in context:
            self._validate_relationships(data, result, context['relationships'])
        
        # Validate temporal consistency
        if 'temporal_field' in context:
            self._validate_temporal_consistency(data, result, context)
    
    def _validate_dict(self, data: Dict[str, Any], result: ValidationResult, context: Dict[str, Any]):
        """
        Validate dictionary data.
        
        Args:
            data: Dictionary to validate
            result: Validation result
            context: Validation context
        """
        # Check required fields
        required_fields = context.get('required_fields', [])
        for field in required_fields:
            if field not in data:
                result.add_error(
                    f"Missing required field: {field}",
                    severity=ValidationSeverity.ERROR
                )
        
        # Check field types
        field_types = context.get('field_types', {})
        for field, expected_type in field_types.items():
            if field in data and not isinstance(data[field], expected_type):
                result.add_error(
                    f"Invalid type for field '{field}': expected {expected_type.__name__}, got {type(data[field]).__name__}",
                    path=field,
                    severity=ValidationSeverity.ERROR
                )
    
    def _validate_list(self, data: List[Any], result: ValidationResult, context: Dict[str, Any]):
        """
        Validate list data.
        
        Args:
            data: List to validate
            result: Validation result
            context: Validation context
        """
        # Check list length
        min_length = context.get('min_length')
        if min_length is not None and len(data) < min_length:
            result.add_error(
                f"List length ({len(data)}) is less than minimum length ({min_length})",
                severity=ValidationSeverity.ERROR
            )
        
        max_length = context.get('max_length')
        if max_length is not None and len(data) > max_length:
            result.add_error(
                f"List length ({len(data)}) is greater than maximum length ({max_length})",
                severity=ValidationSeverity.ERROR
            )
        
        # Check item type
        item_type = context.get('item_type')
        if item_type:
            for i, item in enumerate(data):
                if not isinstance(item, item_type):
                    result.add_error(
                        f"Invalid type for item at index {i}: expected {item_type.__name__}, got {type(item).__name__}",
                        path=f"[{i}]",
                        severity=ValidationSeverity.ERROR
                    )
    
    def _validate_numeric(self, data: Union[int, float], result: ValidationResult, context: Dict[str, Any]):
        """
        Validate numeric data.
        
        Args:
            data: Numeric value to validate
            result: Validation result
            context: Validation context
        """
        # Check minimum value
        min_value = context.get('min_value')
        if min_value is not None and data < min_value:
            result.add_error(
                f"Value ({data}) is less than minimum value ({min_value})",
                severity=ValidationSeverity.ERROR
            )
        
        # Check maximum value
        max_value = context.get('max_value')
        if max_value is not None and data > max_value:
            result.add_error(
                f"Value ({data}) is greater than maximum value ({max_value})",
                severity=ValidationSeverity.ERROR
            )
        
        # Check for NaN or infinity
        if isinstance(data, float) and (np.isnan(data) or np.isinf(data)):
            result.add_error(
                f"Invalid numeric value: {'NaN' if np.isnan(data) else 'Infinity'}",
                severity=ValidationSeverity.ERROR
            )
    
    def _validate_string(self, data: str, result: ValidationResult, context: Dict[str, Any]):
        """
        Validate string data.
        
        Args:
            data: String to validate
            result: Validation result
            context: Validation context
        """
        # Check string length
        min_length = context.get('min_length')
        if min_length is not None and len(data) < min_length:
            result.add_error(
                f"String length ({len(data)}) is less than minimum length ({min_length})",
                severity=ValidationSeverity.ERROR
            )
        
        max_length = context.get('max_length')
        if max_length is not None and len(data) > max_length:
            result.add_error(
                f"String length ({len(data)}) is greater than maximum length ({max_length})",
                severity=ValidationSeverity.ERROR
            )
        
        # Check pattern
        pattern = context.get('pattern')
        if pattern and not re.match(pattern, data):
            result.add_error(
                f"String does not match pattern: {pattern}",
                severity=ValidationSeverity.ERROR
            )
        
        # Check enum values
        enum_values = context.get('enum')
        if enum_values and data not in enum_values:
            result.add_error(
                f"String value '{data}' is not one of the allowed values: {enum_values}",
                severity=ValidationSeverity.ERROR
            )
    
    def _validate_datetime(self, data: datetime, result: ValidationResult, context: Dict[str, Any]):
        """
        Validate datetime data.
        
        Args:
            data: Datetime to validate
            result: Validation result
            context: Validation context
        """
        # Check minimum datetime
        min_datetime = context.get('min_datetime')
        if min_datetime and data < min_datetime:
            result.add_error(
                f"Datetime ({data.isoformat()}) is earlier than minimum datetime ({min_datetime.isoformat()})",
                severity=ValidationSeverity.ERROR
            )
        
        # Check maximum datetime
        max_datetime = context.get('max_datetime')
        if max_datetime and data > max_datetime:
            result.add_error(
                f"Datetime ({data.isoformat()}) is later than maximum datetime ({max_datetime.isoformat()})",
                severity=ValidationSeverity.ERROR
            )
    
    def _validate_dataframe(self, data: pd.DataFrame, result: ValidationResult, context: Dict[str, Any]):
        """
        Validate pandas DataFrame.
        
        Args:
            data: DataFrame to validate
            result: Validation result
            context: Validation context
        """
        # Check required columns
        required_columns = context.get('required_columns', [])
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            result.add_error(
                f"Missing required columns: {missing_columns}",
                severity=ValidationSeverity.ERROR
            )
        
        # Check for empty DataFrame
        if data.empty:
            result.add_warning("DataFrame is empty")
        
        # Check for duplicate rows
        if context.get('check_duplicates', False) and data.duplicated().any():
            result.add_error(
                f"DataFrame contains {data.duplicated().sum()} duplicate rows",
                severity=ValidationSeverity.ERROR
            )
        
        # Check for missing values
        if context.get('check_missing', False):
            missing_counts = data.isnull().sum()
            columns_with_missing = missing_counts[missing_counts > 0]
            if not columns_with_missing.empty:
                for col, count in columns_with_missing.items():
                    result.add_warning(
                        f"Column '{col}' has {count} missing values",
                        path=col
                    )
    
    def _detect_anomalies(self, data: Any, data_key: str, result: ValidationResult, context: Dict[str, Any]):
        """
        Detect anomalies in data.
        
        Args:
            data: Data to check for anomalies
            data_key: Data key for historical data
            result: Validation result
            context: Validation context
        """
        historical_data = self.historical_data[data_key]
        
        # Skip anomaly detection if historical data is not suitable
        if historical_data is None:
            return
        
        # Detect anomalies in numeric data
        if isinstance(data, (int, float)) and isinstance(historical_data, (list, np.ndarray)):
            self._detect_numeric_anomalies(data, historical_data, result, context)
        
        # Detect anomalies in dictionary data
        elif isinstance(data, dict) and isinstance(historical_data, list) and all(isinstance(h, dict) for h in historical_data):
            self._detect_dict_anomalies(data, historical_data, result, context)
    
    def _detect_numeric_anomalies(self, data: Union[int, float], historical_data: Union[List, np.ndarray], result: ValidationResult, context: Dict[str, Any]):
        """
        Detect anomalies in numeric data.
        
        Args:
            data: Numeric value to check
            historical_data: Historical numeric values
            result: Validation result
            context: Validation context
        """
        # Convert historical data to numpy array
        hist_array = np.array(historical_data)
        
        # Calculate statistics
        mean = np.mean(hist_array)
        std = np.std(hist_array)
        
        # Z-score anomaly detection
        z_score = (data - mean) / std if std > 0 else 0
        z_threshold = context.get('z_threshold', 3.0)
        
        if abs(z_score) > z_threshold:
            result.add_warning(
                f"Anomaly detected: value {data} is {abs(z_score):.2f} standard deviations from the mean {mean:.2f}",
                details={
                    'z_score': z_score,
                    'mean': mean,
                    'std': std,
                    'threshold': z_threshold
                }
            )
    
    def _detect_dict_anomalies(self, data: Dict[str, Any], historical_data: List[Dict[str, Any]], result: ValidationResult, context: Dict[str, Any]):
        """
        Detect anomalies in dictionary data.
        
        Args:
            data: Dictionary to check
            historical_data: Historical dictionaries
            result: Validation result
            context: Validation context
        """
        # Check for anomalies in numeric fields
        for key, value in data.items():
            if isinstance(value, (int, float)):
                # Extract historical values for this field
                hist_values = [h.get(key) for h in historical_data if key in h and isinstance(h[key], (int, float))]
                
                # Skip if not enough historical data
                if len(hist_values) < 2:
                    continue
                
                # Detect anomalies
                self._detect_numeric_anomalies(value, hist_values, result, context)
                result.warnings[-1]['path'] = key  # Add field path to warning
    
    def _validate_relationships(self, data: Dict[str, Any], result: ValidationResult, relationships: List[Dict[str, Any]]):
        """
        Validate relationships between fields.
        
        Args:
            data: Dictionary to validate
            result: Validation result
            relationships: List of relationship definitions
        """
        for relationship in relationships:
            rel_type = relationship.get('type')
            fields = relationship.get('fields', [])
            
            # Skip if missing required information
            if not rel_type or not fields or len(fields) < 2:
                continue
            
            # Check if all fields exist
            if not all(field in data for field in fields):
                continue
            
            # Get field values
            values = [data[field] for field in fields]
            
            # Validate based on relationship type
            if rel_type == 'equality':
                if len(set(values)) > 1:
                    result.add_error(
                        f"Equality relationship violated: fields {fields} have different values {values}",
                        severity=ValidationSeverity.ERROR
                    )
            
            elif rel_type == 'inequality':
                if len(set(values)) < len(values):
                    result.add_error(
                        f"Inequality relationship violated: fields {fields} have duplicate values",
                        severity=ValidationSeverity.ERROR
                    )
            
            elif rel_type == 'greater_than':
                if not all(values[i] > values[i+1] for i in range(len(values)-1)):
                    result.add_error(
                        f"Greater-than relationship violated: values {values} for fields {fields} are not in descending order",
                        severity=ValidationSeverity.ERROR
                    )
            
            elif rel_type == 'less_than':
                if not all(values[i] < values[i+1] for i in range(len(values)-1)):
                    result.add_error(
                        f"Less-than relationship violated: values {values} for fields {fields} are not in ascending order",
                        severity=ValidationSeverity.ERROR
                    )
    
    def _validate_temporal_consistency(self, data: Any, result: ValidationResult, context: Dict[str, Any]):
        """
        Validate temporal consistency.
        
        Args:
            data: Data to validate
            result: Validation result
            context: Validation context
        """
        temporal_field = context.get('temporal_field')
        if not temporal_field:
            return
        
        # Handle dictionary data
        if isinstance(data, dict) and temporal_field in data:
            temporal_value = data[temporal_field]
            
            # Check if temporal value is a datetime
            if not isinstance(temporal_value, datetime):
                try:
                    # Try to parse as datetime
                    temporal_value = datetime.fromisoformat(temporal_value)
                except (ValueError, TypeError):
                    result.add_error(
                        f"Invalid temporal value for field '{temporal_field}': {temporal_value}",
                        path=temporal_field,
                        severity=ValidationSeverity.ERROR
                    )
                    return
            
            # Check if temporal value is in the future
            if temporal_value > datetime.now() and not context.get('allow_future', False):
                result.add_error(
                    f"Temporal value for field '{temporal_field}' is in the future: {temporal_value.isoformat()}",
                    path=temporal_field,
                    severity=ValidationSeverity.ERROR
                )
            
            # Check if temporal value is too old
            max_age = context.get('max_age')
            if max_age and (datetime.now() - temporal_value).total_seconds() > max_age:
                result.add_warning(
                    f"Temporal value for field '{temporal_field}' is older than maximum age: {temporal_value.isoformat()}",
                    path=temporal_field
                )
    
    def _apply_validation_rules(self, data: Any, result: ValidationResult, context: Dict[str, Any]):
        """
        Apply custom validation rules.
        
        Args:
            data: Data to validate
            result: Validation result
            context: Validation context
        """
        for rule in self.validation_rules:
            try:
                rule_result = rule['func'](data, context)
                
                # Handle boolean result
                if isinstance(rule_result, bool):
                    if not rule_result:
                        result.add_error(
                            f"Validation rule '{rule['name']}' failed",
                            severity=ValidationSeverity.ERROR
                        )
                
                # Handle dictionary result
                elif isinstance(rule_result, dict):
                    is_valid = rule_result.get('is_valid', True)
                    message = rule_result.get('message', f"Validation rule '{rule['name']}' failed")
                    path = rule_result.get('path')
                    severity_name = rule_result.get('severity', 'ERROR')
                    details = rule_result.get('details')
                    
                    # Map severity name to enum
                    severity = ValidationSeverity.ERROR
                    try:
                        severity = ValidationSeverity[severity_name]
                    except (KeyError, TypeError):
                        pass
                    
                    if not is_valid:
                        result.add_error(
                            message=message,
                            path=path,
                            severity=severity,
                            details=details
                        )
                
                # Handle ValidationResult
                elif isinstance(rule_result, ValidationResult):
                    # Merge errors and warnings
                    for error in rule_result.errors:
                        result.errors.append(error)
                    
                    for warning in rule_result.warnings:
                        result.warnings.append(warning)
                    
                    # Update validity
                    if not rule_result.is_valid:
                        result.is_valid = False
            
            except Exception as e:
                logger.error(f"Error applying validation rule '{rule['name']}': {e}")
                result.add_error(
                    f"Error applying validation rule '{rule['name']}': {str(e)}",
                    severity=ValidationSeverity.ERROR
                )

# Create default validator instance
default_validator = DataValidator()