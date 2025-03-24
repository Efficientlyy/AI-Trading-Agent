"""
Pydantic compatibility layer for Python 3.13

This module provides compatibility fixes for pydantic and annotated_types
when using Python 3.13, which has incompatible changes with older packages.
"""

import sys
import inspect
import importlib
from typing import Any, Dict, List, Optional, Union, get_type_hints, get_origin, get_args

# Create a mock for BaseMetadata from annotated_types if it can't be imported
class MockBaseMetadata:
    """Mock implementation of BaseMetadata from annotated_types."""
    
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

# ValidationInfo compatibility class
class ValidationInfo:
    """Mock implementation of ValidationInfo for pydantic compatibility."""
    
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        
    def __call__(self, *args, **kwargs):
        return self

# Try to provide annotated_types.BaseMetadata
try:
    from annotated_types import BaseMetadata
except ImportError:
    try:
        # Try to patch the module
        if 'annotated_types' in sys.modules:
            sys.modules['annotated_types'].BaseMetadata = MockBaseMetadata
        else:
            # Create a mock module
            class MockAnnotatedTypes:
                BaseMetadata = MockBaseMetadata
            
            sys.modules["annotated_types"] = MockAnnotatedTypes()
        
        # Now we can import it
        from annotated_types import BaseMetadata
    except ImportError:
        # If all else fails, just use our mock
        BaseMetadata = MockBaseMetadata

# Field function for pydantic compatibility
def Field(default=None, **kwargs):
    """Compatibility implementation of pydantic.Field."""
    return default

# Field validator compatibility function
def field_validator(field_name, *args, **kwargs):
    """Compatibility implementation of pydantic.field_validator."""
    def decorator(func):
        func._validator_info = {
            'field_name': field_name,
            'args': args,
            'kwargs': kwargs,
            'type': 'field_validator'
        }
        return func
    return decorator

# Try to provide other commonly needed compatibility for Pydantic
try:
    from pydantic import (
        BaseModel, 
        Field as PydanticField, 
        ValidationInfo as PydanticValidationInfo,
        field_validator as pydantic_field_validator
    )
    Field = PydanticField
    ValidationInfo = PydanticValidationInfo
    field_validator = pydantic_field_validator
except ImportError:
    # Create a basic BaseModel if pydantic is not available
    class BaseModel:
        """Mock implementation of BaseModel for when pydantic is unavailable."""
        
        def __init__(self, **data):
            for key, value in data.items():
                setattr(self, key, value)
        
        @classmethod
        def model_validate(cls, data, **kwargs):
            """Mock validation method."""
            return cls(**data)
        
        def model_dump(self):
            """Mock dump method."""
            return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

# Function to create a model without relying on pydantic directly
def create_model_compat(name, **field_definitions):
    """Create a model class compatible with both old and new pydantic."""
    try:
        # Try the pydantic way first
        from src.common.pydantic_compat import create_model_compat as create_model
        return create_model(name, **field_definitions)
    except ImportError:
        # Fall back to a basic implementation
        attrs = {
            '__annotations__': {name: type_ for name, (type_, _) in field_definitions.items()},
            **{name: default for name, (_, default) in field_definitions.items()}
        }
        return type(name, (BaseModel,), attrs)

# Validator function that works with or without pydantic
def validator_compat(field_name, *args, **kwargs):
    """A compatibility layer for validators that works with or without pydantic."""
    def decorator(func):
        func._validator_info = {
            'field_name': field_name,
            'args': args,
            'kwargs': kwargs
        }
        return func
    
    return decorator
