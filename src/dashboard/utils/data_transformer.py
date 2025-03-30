"""
Data Transformer

This module provides a flexible data transformation pipeline for normalizing,
enriching, and optimizing data from various sources before presentation.
"""

import copy
import json
import logging
import re
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("data_transformer")

class TransformationStage(Enum):
    """Transformation stage enum"""
    NORMALIZATION = 1
    STANDARDIZATION = 2
    ENRICHMENT = 3
    OPTIMIZATION = 4

class TransformationResult:
    """Transformation result class"""
    
    def __init__(self, 
                 data: Any,
                 original_data: Any = None,
                 metadata: Dict[str, Any] = None):
        """
        Initialize transformation result.
        
        Args:
            data: The transformed data
            original_data: The original data before transformation
            metadata: Metadata about the transformation
        """
        self.data = data
        self.original_data = original_data if original_data is not None else copy.deepcopy(data)
        self.metadata = metadata or {}
        self.transformation_steps = []
        self.timestamp = datetime.now().isoformat()
    
    def add_step(self, 
                 step_name: str, 
                 stage: TransformationStage,
                 details: Dict[str, Any] = None):
        """
        Add a transformation step to the result.
        
        Args:
            step_name: The name of the transformation step
            stage: The transformation stage
            details: Additional details about the step
        """
        self.transformation_steps.append({
            'step_name': step_name,
            'stage': stage.name,
            'details': details or {},
            'timestamp': datetime.now().isoformat()
        })
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert transformation result to dictionary.
        
        Returns:
            Dictionary representation of transformation result
        """
        return {
            'metadata': self.metadata,
            'transformation_steps': self.transformation_steps,
            'timestamp': self.timestamp
        }
    
    def __str__(self) -> str:
        """
        String representation of transformation result.
        
        Returns:
            String representation
        """
        return json.dumps(self.to_dict(), indent=2)

class TransformationStep:
    """Transformation step class"""
    
    def __init__(self, 
                 name: str,
                 transform_func: Callable,
                 stage: TransformationStage,
                 config: Dict[str, Any] = None):
        """
        Initialize transformation step.
        
        Args:
            name: The name of the transformation step
            transform_func: The transformation function
            stage: The transformation stage
            config: Configuration for the transformation
        """
        self.name = name
        self.transform_func = transform_func
        self.stage = stage
        self.config = config or {}
    
    def apply(self, 
              data: Any, 
              result: TransformationResult) -> Any:
        """
        Apply the transformation step to the data.
        
        Args:
            data: The data to transform
            result: The transformation result
            
        Returns:
            The transformed data
        """
        try:
            # Apply transformation
            transformed_data = self.transform_func(data, self.config)
            
            # Add step to result
            result.add_step(
                step_name=self.name,
                stage=self.stage,
                details=self.config
            )
            
            return transformed_data
        except Exception as e:
            logger.error(f"Error applying transformation step '{self.name}': {e}")
            
            # Add error details to result
            result.add_step(
                step_name=self.name,
                stage=self.stage,
                details={
                    'error': str(e),
                    'config': self.config
                }
            )
            
            # Return original data
            return data

class DataTransformer:
    """
    Flexible data transformation pipeline for normalizing, enriching, and optimizing data.
    """
    
    def __init__(self, name: str = "default"):
        """
        Initialize data transformer.
        
        Args:
            name: The name of the transformer
        """
        self.name = name
        self.steps = []
        self.enabled = True
        
        logger.info(f"Data transformer '{name}' initialized")
    
    def add_step(self, 
                 name: str,
                 transform_func: Callable,
                 stage: TransformationStage,
                 config: Dict[str, Any] = None) -> 'DataTransformer':
        """
        Add a transformation step to the pipeline.
        
        Args:
            name: The name of the transformation step
            transform_func: The transformation function
            stage: The transformation stage
            config: Configuration for the transformation
            
        Returns:
            The data transformer instance (for chaining)
        """
        step = TransformationStep(
            name=name,
            transform_func=transform_func,
            stage=stage,
            config=config
        )
        
        self.steps.append(step)
        logger.info(f"Added transformation step '{name}' to transformer '{self.name}'")
        
        return self
    
    def transform(self, 
                  data: Any, 
                  metadata: Dict[str, Any] = None) -> TransformationResult:
        """
        Transform data using the pipeline.
        
        Args:
            data: The data to transform
            metadata: Metadata about the data
            
        Returns:
            The transformation result
        """
        if not self.enabled:
            logger.info(f"Transformer '{self.name}' is disabled, skipping transformation")
            return TransformationResult(data=data, metadata=metadata)
        
        # Initialize result
        result = TransformationResult(
            data=data,
            original_data=data,
            metadata=metadata
        )
        
        # Apply each step in the pipeline
        transformed_data = data
        
        for step in self.steps:
            transformed_data = step.apply(transformed_data, result)
        
        # Update result with final data
        result.data = transformed_data
        
        logger.info(f"Transformation completed for transformer '{self.name}' with {len(self.steps)} steps")
        
        return result
    
    def enable(self):
        """Enable the transformer"""
        self.enabled = True
        logger.info(f"Transformer '{self.name}' enabled")
    
    def disable(self):
        """Disable the transformer"""
        self.enabled = False
        logger.info(f"Transformer '{self.name}' disabled")
    
    def clear_steps(self):
        """Clear all transformation steps"""
        self.steps = []
        logger.info(f"Cleared all transformation steps from transformer '{self.name}'")
    
    def get_steps_by_stage(self, stage: TransformationStage) -> List[TransformationStep]:
        """
        Get transformation steps by stage.
        
        Args:
            stage: The transformation stage
            
        Returns:
            List of transformation steps
        """
        return [step for step in self.steps if step.stage == stage]

# Built-in transformation functions

def normalize_field_names(data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize field names in a dictionary.
    
    Args:
        data: The dictionary to normalize
        config: Configuration for normalization
            - field_mapping: Dictionary mapping original field names to normalized names
            - case: Case to convert field names to (lower, upper, none)
            - remove_spaces: Whether to remove spaces from field names
            - separator: Character to replace spaces with
            
    Returns:
        Dictionary with normalized field names
    """
    if not isinstance(data, dict):
        return data
    
    # Get configuration
    field_mapping = config.get('field_mapping', {})
    case = config.get('case', 'lower')
    remove_spaces = config.get('remove_spaces', True)
    separator = config.get('separator', '_')
    
    # Create normalized dictionary
    normalized = {}
    
    for key, value in data.items():
        # Apply field mapping if available
        if key in field_mapping:
            normalized_key = field_mapping[key]
        else:
            normalized_key = key
            
            # Apply case conversion
            if case == 'lower':
                normalized_key = normalized_key.lower()
            elif case == 'upper':
                normalized_key = normalized_key.upper()
            
            # Remove or replace spaces
            if remove_spaces:
                normalized_key = normalized_key.replace(' ', separator)
        
        # Add to normalized dictionary
        normalized[normalized_key] = value
    
    return normalized

def normalize_numeric_values(data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize numeric values in a dictionary.
    
    Args:
        data: The dictionary to normalize
        config: Configuration for normalization
            - fields: List of fields to normalize
            - method: Normalization method (min_max, z_score, decimal_places)
            - min_value: Minimum value for min_max normalization
            - max_value: Maximum value for min_max normalization
            - decimal_places: Number of decimal places for rounding
            
    Returns:
        Dictionary with normalized numeric values
    """
    if not isinstance(data, dict):
        return data
    
    # Get configuration
    fields = config.get('fields', [])
    method = config.get('method', 'min_max')
    min_value = config.get('min_value')
    max_value = config.get('max_value')
    decimal_places = config.get('decimal_places')
    
    # Create normalized dictionary
    normalized = copy.deepcopy(data)
    
    for field in fields:
        if field in normalized and isinstance(normalized[field], (int, float)):
            value = normalized[field]
            
            # Apply normalization method
            if method == 'min_max' and min_value is not None and max_value is not None:
                # Min-max normalization
                normalized[field] = (value - min_value) / (max_value - min_value)
            elif method == 'z_score' and 'mean' in config and 'std' in config:
                # Z-score normalization
                normalized[field] = (value - config['mean']) / config['std']
            elif method == 'decimal_places' and decimal_places is not None:
                # Round to decimal places
                normalized[field] = round(value, decimal_places)
    
    return normalized

def standardize_datetime_format(data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Standardize datetime format in a dictionary.
    
    Args:
        data: The dictionary to standardize
        config: Configuration for standardization
            - fields: List of fields to standardize
            - input_format: Input datetime format
            - output_format: Output datetime format
            - timezone: Timezone to convert to
            
    Returns:
        Dictionary with standardized datetime format
    """
    if not isinstance(data, dict):
        return data
    
    # Get configuration
    fields = config.get('fields', [])
    input_format = config.get('input_format')
    output_format = config.get('output_format', '%Y-%m-%dT%H:%M:%S.%fZ')
    timezone_name = config.get('timezone')
    
    # Create standardized dictionary
    standardized = copy.deepcopy(data)
    
    for field in fields:
        if field in standardized and isinstance(standardized[field], str):
            value = standardized[field]
            
            try:
                # Parse datetime
                if input_format:
                    dt = datetime.strptime(value, input_format)
                else:
                    # Try common formats
                    for fmt in ['%Y-%m-%dT%H:%M:%S.%fZ', '%Y-%m-%d %H:%M:%S', '%Y-%m-%d']:
                        try:
                            dt = datetime.strptime(value, fmt)
                            break
                        except ValueError:
                            continue
                    else:
                        # If no format works, try parsing with dateutil
                        from dateutil import parser
                        dt = parser.parse(value)
                
                # Convert timezone if specified
                if timezone_name:
                    import pytz
                    timezone = pytz.timezone(timezone_name)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone)
                    else:
                        dt = dt.astimezone(timezone)
                
                # Format datetime
                standardized[field] = dt.strftime(output_format)
            except Exception as e:
                logger.warning(f"Error standardizing datetime for field '{field}': {e}")
    
    return standardized

def enrich_with_calculated_fields(data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Enrich data with calculated fields.
    
    Args:
        data: The dictionary to enrich
        config: Configuration for enrichment
            - calculations: Dictionary mapping new field names to calculation expressions
            
    Returns:
        Dictionary with calculated fields
    """
    if not isinstance(data, dict):
        return data
    
    # Get configuration
    calculations = config.get('calculations', {})
    
    # Create enriched dictionary
    enriched = copy.deepcopy(data)
    
    for new_field, expression in calculations.items():
        try:
            # Create a local namespace with the data
            namespace = {**data}
            
            # Add numpy functions to namespace
            for name in dir(np):
                if not name.startswith('_'):
                    namespace[name] = getattr(np, name)
            
            # Evaluate expression
            result = eval(expression, {"__builtins__": {}}, namespace)
            
            # Add calculated field
            enriched[new_field] = result
        except Exception as e:
            logger.warning(f"Error calculating field '{new_field}': {e}")
    
    return enriched

def optimize_data_size(data: Any, config: Dict[str, Any]) -> Any:
    """
    Optimize data size by removing unnecessary fields or downsampling.
    
    Args:
        data: The data to optimize
        config: Configuration for optimization
            - remove_fields: List of fields to remove
            - keep_fields: List of fields to keep (all others will be removed)
            - downsample: Whether to downsample time series data
            - downsample_factor: Factor by which to downsample
            
    Returns:
        Optimized data
    """
    # Handle dictionary data
    if isinstance(data, dict):
        # Get configuration
        remove_fields = config.get('remove_fields', [])
        keep_fields = config.get('keep_fields')
        
        # Create optimized dictionary
        if keep_fields:
            # Keep only specified fields
            optimized = {k: v for k, v in data.items() if k in keep_fields}
        else:
            # Remove specified fields
            optimized = {k: v for k, v in data.items() if k not in remove_fields}
        
        return optimized
    
    # Handle list data
    elif isinstance(data, list):
        # Get configuration
        downsample = config.get('downsample', False)
        downsample_factor = config.get('downsample_factor', 2)
        
        # Downsample if specified
        if downsample and len(data) > downsample_factor:
            # Simple downsampling by taking every nth element
            return data[::downsample_factor]
        
        return data
    
    # Handle pandas DataFrame
    elif isinstance(data, pd.DataFrame):
        # Get configuration
        remove_fields = config.get('remove_fields', [])
        keep_fields = config.get('keep_fields')
        downsample = config.get('downsample', False)
        downsample_factor = config.get('downsample_factor', 2)
        
        # Create optimized DataFrame
        if keep_fields:
            # Keep only specified columns
            optimized = data[keep_fields].copy()
        else:
            # Remove specified columns
            optimized = data.drop(columns=remove_fields, errors='ignore')
        
        # Downsample if specified
        if downsample and len(optimized) > downsample_factor:
            # Simple downsampling by taking every nth row
            optimized = optimized.iloc[::downsample_factor]
        
        return optimized
    
    # Return original data for other types
    return data

# Create transformer registry
transformer_registry = {}

def get_transformer(name: str = "default") -> DataTransformer:
    """
    Get a transformer by name, creating it if it doesn't exist.
    
    Args:
        name: The name of the transformer
        
    Returns:
        The data transformer
    """
    if name not in transformer_registry:
        transformer_registry[name] = DataTransformer(name)
    
    return transformer_registry[name]

def register_transformer(transformer: DataTransformer):
    """
    Register a transformer in the registry.
    
    Args:
        transformer: The transformer to register
    """
    transformer_registry[transformer.name] = transformer
    logger.info(f"Registered transformer '{transformer.name}'")

def create_default_transformers():
    """Create default transformers for common data types"""
    # Market data transformer
    market_transformer = DataTransformer("market_data")
    market_transformer.add_step(
        name="normalize_field_names",
        transform_func=normalize_field_names,
        stage=TransformationStage.NORMALIZATION,
        config={
            "field_mapping": {
                "s": "symbol",
                "p": "price",
                "v": "volume",
                "t": "timestamp"
            },
            "case": "lower",
            "remove_spaces": True
        }
    ).add_step(
        name="standardize_datetime",
        transform_func=standardize_datetime_format,
        stage=TransformationStage.STANDARDIZATION,
        config={
            "fields": ["timestamp"],
            "output_format": "%Y-%m-%dT%H:%M:%S.%fZ",
            "timezone": "UTC"
        }
    ).add_step(
        name="add_calculated_fields",
        transform_func=enrich_with_calculated_fields,
        stage=TransformationStage.ENRICHMENT,
        config={
            "calculations": {
                "value": "price * volume",
                "price_change_pct": "0.0"  # Placeholder, would be calculated with historical data
            }
        }
    )
    register_transformer(market_transformer)
    
    # Trade data transformer
    trade_transformer = DataTransformer("trade_data")
    trade_transformer.add_step(
        name="normalize_field_names",
        transform_func=normalize_field_names,
        stage=TransformationStage.NORMALIZATION,
        config={
            "field_mapping": {
                "sym": "symbol",
                "px": "price",
                "qty": "quantity",
                "side": "direction",
                "ts": "timestamp"
            },
            "case": "lower",
            "remove_spaces": True
        }
    ).add_step(
        name="standardize_datetime",
        transform_func=standardize_datetime_format,
        stage=TransformationStage.STANDARDIZATION,
        config={
            "fields": ["timestamp"],
            "output_format": "%Y-%m-%dT%H:%M:%S.%fZ",
            "timezone": "UTC"
        }
    ).add_step(
        name="add_calculated_fields",
        transform_func=enrich_with_calculated_fields,
        stage=TransformationStage.ENRICHMENT,
        config={
            "calculations": {
                "value": "price * quantity",
                "fee": "price * quantity * 0.001"  # Example fee calculation
            }
        }
    )
    register_transformer(trade_transformer)
    
    # Position data transformer
    position_transformer = DataTransformer("position_data")
    position_transformer.add_step(
        name="normalize_field_names",
        transform_func=normalize_field_names,
        stage=TransformationStage.NORMALIZATION,
        config={
            "field_mapping": {
                "sym": "symbol",
                "pos": "position",
                "entry_px": "entry_price",
                "curr_px": "current_price",
                "ts": "timestamp"
            },
            "case": "lower",
            "remove_spaces": True
        }
    ).add_step(
        name="add_calculated_fields",
        transform_func=enrich_with_calculated_fields,
        stage=TransformationStage.ENRICHMENT,
        config={
            "calculations": {
                "value": "position * current_price",
                "pnl": "position * (current_price - entry_price)",
                "pnl_pct": "(current_price - entry_price) / entry_price * 100"
            }
        }
    )
    register_transformer(position_transformer)

# Initialize default transformers
create_default_transformers()