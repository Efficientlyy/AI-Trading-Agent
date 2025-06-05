#!/usr/bin/env python
"""
ConfigRegistry Module for Modular System Overseer

This module implements a centralized configuration registry with extension points
for modular parameter management. It supports dynamic parameter registration,
validation, grouping, and persistence.
"""

import os
import json
import yaml
import logging
import threading
from typing import Any, Dict, List, Optional, Callable, Union
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("config_registry")

class ParameterDefinition:
    """Definition of a configuration parameter"""
    
    def __init__(
        self,
        module_id: str,
        param_id: str,
        default_value: Any,
        param_type: type = None,
        description: str = None,
        group: str = None,
        tags: List[str] = None,
        validation_func: Callable[[Any], bool] = None,
        options: List[Any] = None,
        min_value: Any = None,
        max_value: Any = None,
        secret: bool = False,
        restart_required: bool = False,
        deprecated: bool = False,
        experimental: bool = False
    ):
        """Initialize parameter definition
        
        Args:
            module_id: ID of the module that owns this parameter
            param_id: ID of the parameter
            default_value: Default value for the parameter
            param_type: Expected type of the parameter
            description: Description of the parameter
            group: Group this parameter belongs to
            tags: List of tags for categorizing the parameter
            validation_func: Function to validate parameter values
            options: List of valid options for this parameter
            min_value: Minimum allowed value
            max_value: Maximum allowed value
            secret: Whether this parameter contains sensitive information
            restart_required: Whether changes require system restart
            deprecated: Whether this parameter is deprecated
            experimental: Whether this parameter is experimental
        """
        self.module_id = module_id
        self.param_id = param_id
        self.default_value = default_value
        self.param_type = param_type or type(default_value)
        self.description = description
        self.group = group
        self.tags = tags or []
        self.validation_func = validation_func
        self.options = options
        self.min_value = min_value
        self.max_value = max_value
        self.secret = secret
        self.restart_required = restart_required
        self.deprecated = deprecated
        self.experimental = experimental
        
    def validate(self, value: Any) -> bool:
        """Validate a parameter value
        
        Args:
            value: Value to validate
            
        Returns:
            bool: True if valid, False otherwise
        """
        # Check type if specified
        if self.param_type and not isinstance(value, self.param_type):
            try:
                # Attempt type conversion
                value = self.param_type(value)
            except (ValueError, TypeError):
                logger.error(f"Type validation failed for {self.module_id}.{self.param_id}: "
                            f"expected {self.param_type.__name__}, got {type(value).__name__}")
                return False
        
        # Check options if specified
        if self.options is not None and value not in self.options:
            logger.error(f"Option validation failed for {self.module_id}.{self.param_id}: "
                        f"value {value} not in options {self.options}")
            return False
        
        # Check min/max if specified and value is comparable
        if self.min_value is not None and value < self.min_value:
            logger.error(f"Min validation failed for {self.module_id}.{self.param_id}: "
                        f"value {value} less than minimum {self.min_value}")
            return False
            
        if self.max_value is not None and value > self.max_value:
            logger.error(f"Max validation failed for {self.module_id}.{self.param_id}: "
                        f"value {value} greater than maximum {self.max_value}")
            return False
        
        # Run custom validation function if specified
        if self.validation_func and not self.validation_func(value):
            logger.error(f"Custom validation failed for {self.module_id}.{self.param_id}")
            return False
            
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert parameter definition to dictionary
        
        Returns:
            dict: Dictionary representation of parameter definition
        """
        return {
            "module_id": self.module_id,
            "param_id": self.param_id,
            "default_value": self.default_value if not self.secret else "********",
            "param_type": self.param_type.__name__ if self.param_type else None,
            "description": self.description,
            "group": self.group,
            "tags": self.tags,
            "options": self.options,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "secret": self.secret,
            "restart_required": self.restart_required,
            "deprecated": self.deprecated,
            "experimental": self.experimental
        }


class ParameterChange:
    """Record of a parameter change"""
    
    def __init__(
        self,
        module_id: str,
        param_id: str,
        old_value: Any,
        new_value: Any,
        timestamp: float = None,
        user_id: str = None
    ):
        """Initialize parameter change record
        
        Args:
            module_id: ID of the module that owns this parameter
            param_id: ID of the parameter
            old_value: Previous value
            new_value: New value
            timestamp: Time of change (defaults to current time)
            user_id: ID of user who made the change
        """
        self.module_id = module_id
        self.param_id = param_id
        self.old_value = old_value
        self.new_value = new_value
        self.timestamp = timestamp or datetime.now().timestamp()
        self.user_id = user_id
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert parameter change to dictionary
        
        Returns:
            dict: Dictionary representation of parameter change
        """
        return {
            "module_id": self.module_id,
            "param_id": self.param_id,
            "old_value": self.old_value,
            "new_value": self.new_value,
            "timestamp": self.timestamp,
            "user_id": self.user_id
        }


class ParameterGroup:
    """Group of related parameters"""
    
    def __init__(
        self,
        group_id: str,
        name: str = None,
        description: str = None,
        parent_group: str = None
    ):
        """Initialize parameter group
        
        Args:
            group_id: ID of the group
            name: Display name of the group
            description: Description of the group
            parent_group: ID of parent group
        """
        self.group_id = group_id
        self.name = name or group_id
        self.description = description
        self.parent_group = parent_group
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert parameter group to dictionary
        
        Returns:
            dict: Dictionary representation of parameter group
        """
        return {
            "group_id": self.group_id,
            "name": self.name,
            "description": self.description,
            "parent_group": self.parent_group
        }


class ParameterPreset:
    """Preset of parameter values"""
    
    def __init__(
        self,
        preset_id: str,
        name: str = None,
        description: str = None,
        parameters: Dict[str, Dict[str, Any]] = None,
        tags: List[str] = None
    ):
        """Initialize parameter preset
        
        Args:
            preset_id: ID of the preset
            name: Display name of the preset
            description: Description of the preset
            parameters: Dictionary of parameter values by module_id and param_id
            tags: List of tags for categorizing the preset
        """
        self.preset_id = preset_id
        self.name = name or preset_id
        self.description = description
        self.parameters = parameters or {}
        self.tags = tags or []
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert parameter preset to dictionary
        
        Returns:
            dict: Dictionary representation of parameter preset
        """
        return {
            "preset_id": self.preset_id,
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "tags": self.tags
        }


class ConfigRegistry:
    """Centralized configuration registry with extension points"""
    
    def __init__(self, config_dir: str = None, event_bus=None):
        """Initialize configuration registry
        
        Args:
            config_dir: Directory for configuration files
            event_bus: Event bus for publishing configuration events
        """
        self.config_dir = config_dir or os.path.join(os.getcwd(), "config")
        self.event_bus = event_bus
        
        # Create config directory if it doesn't exist
        os.makedirs(self.config_dir, exist_ok=True)
        
        # Parameter definitions and values
        self.parameter_definitions = {}  # {(module_id, param_id): ParameterDefinition}
        self.parameter_values = {}       # {(module_id, param_id): value}
        
        # Parameter groups
        self.parameter_groups = {}       # {group_id: ParameterGroup}
        
        # Parameter presets
        self.parameter_presets = {}      # {preset_id: ParameterPreset}
        
        # Change history
        self.change_history = []         # [ParameterChange]
        
        # Custom validators
        self.validators = {}             # {(module_id, param_id): validation_func}
        
        # Extension points
        self.pre_set_hooks = {}          # {(module_id, param_id): [hook_func]}
        self.post_set_hooks = {}         # {(module_id, param_id): [hook_func]}
        self.pre_load_hooks = []         # [hook_func]
        self.post_load_hooks = []        # [hook_func]
        self.pre_save_hooks = []         # [hook_func]
        self.post_save_hooks = []        # [hook_func]
        
        # Lock for thread safety
        self.lock = threading.RLock()
        
        # Load default configuration
        self.load_config()
        
        logger.info("ConfigRegistry initialized")
    
    def register_parameter(
        self,
        module_id: str,
        param_id: str,
        default_value: Any,
        param_type: type = None,
        description: str = None,
        group: str = None,
        tags: List[str] = None,
        validation_func: Callable[[Any], bool] = None,
        options: List[Any] = None,
        min_value: Any = None,
        max_value: Any = None,
        secret: bool = False,
        restart_required: bool = False,
        deprecated: bool = False,
        experimental: bool = False
    ) -> bool:
        """Register a new parameter
        
        Args:
            module_id: ID of the module that owns this parameter
            param_id: ID of the parameter
            default_value: Default value for the parameter
            param_type: Expected type of the parameter
            description: Description of the parameter
            group: Group this parameter belongs to
            tags: List of tags for categorizing the parameter
            validation_func: Function to validate parameter values
            options: List of valid options for this parameter
            min_value: Minimum allowed value
            max_value: Maximum allowed value
            secret: Whether this parameter contains sensitive information
            restart_required: Whether changes require system restart
            deprecated: Whether this parameter is deprecated
            experimental: Whether this parameter is experimental
            
        Returns:
            bool: True if parameter was registered, False otherwise
        """
        with self.lock:
            param_key = (module_id, param_id)
            
            # Check if parameter already exists
            if param_key in self.parameter_definitions:
                logger.warning(f"Parameter {module_id}.{param_id} already registered")
                return False
            
            # Create parameter definition
            param_def = ParameterDefinition(
                module_id=module_id,
                param_id=param_id,
                default_value=default_value,
                param_type=param_type,
                description=description,
                group=group,
                tags=tags,
                validation_func=validation_func,
                options=options,
                min_value=min_value,
                max_value=max_value,
                secret=secret,
                restart_required=restart_required,
                deprecated=deprecated,
                experimental=experimental
            )
            
            # Store parameter definition
            self.parameter_definitions[param_key] = param_def
            
            # Store custom validator if provided
            if validation_func:
                self.validators[param_key] = validation_func
            
            # Set default value
            if param_key not in self.parameter_values:
                self.parameter_values[param_key] = default_value
            
            # Register parameter group if specified and not already registered
            if group and group not in self.parameter_groups:
                self.register_parameter_group(group)
            
            logger.info(f"Parameter {module_id}.{param_id} registered")
            
            # Publish event if event bus is available
            if self.event_bus:
                self.event_bus.publish(
                    "config.parameter_registered",
                    {
                        "module_id": module_id,
                        "param_id": param_id,
                        "definition": param_def.to_dict()
                    }
                )
            
            return True
    
    def register_parameter_group(
        self,
        group_id: str,
        name: str = None,
        description: str = None,
        parent_group: str = None
    ) -> bool:
        """Register a parameter group
        
        Args:
            group_id: ID of the group
            name: Display name of the group
            description: Description of the group
            parent_group: ID of parent group
            
        Returns:
            bool: True if group was registered, False otherwise
        """
        with self.lock:
            # Check if group already exists
            if group_id in self.parameter_groups:
                logger.warning(f"Parameter group {group_id} already registered")
                return False
            
            # Create parameter group
            group = ParameterGroup(
                group_id=group_id,
                name=name,
                description=description,
                parent_group=parent_group
            )
            
            # Store parameter group
            self.parameter_groups[group_id] = group
            
            # Register parent group if specified and not already registered
            if parent_group and parent_group not in self.parameter_groups:
                self.register_parameter_group(parent_group)
            
            logger.info(f"Parameter group {group_id} registered")
            
            # Publish event if event bus is available
            if self.event_bus:
                self.event_bus.publish(
                    "config.group_registered",
                    {
                        "group_id": group_id,
                        "group": group.to_dict()
                    }
                )
            
            return True
    
    def register_preset(
        self,
        preset_id: str,
        parameters: Dict[str, Dict[str, Any]],
        name: str = None,
        description: str = None,
        tags: List[str] = None
    ) -> bool:
        """Register a parameter preset
        
        Args:
            preset_id: ID of the preset
            parameters: Dictionary of parameter values by module_id and param_id
            name: Display name of the preset
            description: Description of the preset
            tags: List of tags for categorizing the preset
            
        Returns:
            bool: True if preset was registered, False otherwise
        """
        with self.lock:
            # Check if preset already exists
            if preset_id in self.parameter_presets:
                logger.warning(f"Parameter preset {preset_id} already registered")
                return False
            
            # Create parameter preset
            preset = ParameterPreset(
                preset_id=preset_id,
                name=name,
                description=description,
                parameters=parameters,
                tags=tags
            )
            
            # Store parameter preset
            self.parameter_presets[preset_id] = preset
            
            logger.info(f"Parameter preset {preset_id} registered")
            
            # Publish event if event bus is available
            if self.event_bus:
                self.event_bus.publish(
                    "config.preset_registered",
                    {
                        "preset_id": preset_id,
                        "preset": preset.to_dict()
                    }
                )
            
            return True
    
    def get_parameter(
        self,
        module_id: str,
        param_id: str,
        default: Any = None
    ) -> Any:
        """Get parameter value
        
        Args:
            module_id: ID of the module that owns this parameter
            param_id: ID of the parameter
            default: Default value if parameter not found
            
        Returns:
            Any: Parameter value
        """
        with self.lock:
            param_key = (module_id, param_id)
            
            # Return parameter value if it exists
            if param_key in self.parameter_values:
                return self.parameter_values[param_key]
            
            # Return default value from parameter definition if it exists
            if param_key in self.parameter_definitions:
                return self.parameter_definitions[param_key].default_value
            
            # Return provided default value
            return default
    
    def set_parameter(
        self,
        module_id: str,
        param_id: str,
        value: Any,
        user_id: str = None,
        skip_validation: bool = False
    ) -> bool:
        """Set parameter value
        
        Args:
            module_id: ID of the module that owns this parameter
            param_id: ID of the parameter
            value: New parameter value
            user_id: ID of user making the change
            skip_validation: Whether to skip validation
            
        Returns:
            bool: True if parameter was set, False otherwise
        """
        with self.lock:
            param_key = (module_id, param_id)
            
            # Check if parameter exists
            if param_key not in self.parameter_definitions:
                logger.warning(f"Parameter {module_id}.{param_id} not registered")
                return False
            
            # Get current value
            old_value = self.get_parameter(module_id, param_id)
            
            # Run pre-set hooks
            if param_key in self.pre_set_hooks:
                for hook in self.pre_set_hooks[param_key]:
                    try:
                        hook_result = hook(module_id, param_id, old_value, value, user_id)
                        if hook_result is False:
                            logger.warning(f"Pre-set hook rejected change to {module_id}.{param_id}")
                            return False
                        elif hook_result is not None:
                            # Hook modified the value
                            value = hook_result
                    except Exception as e:
                        logger.error(f"Error in pre-set hook for {module_id}.{param_id}: {e}")
            
            # Validate parameter value
            if not skip_validation:
                param_def = self.parameter_definitions[param_key]
                if not param_def.validate(value):
                    logger.warning(f"Validation failed for {module_id}.{param_id}")
                    return False
            
            # Set parameter value
            self.parameter_values[param_key] = value
            
            # Record change
            change = ParameterChange(
                module_id=module_id,
                param_id=param_id,
                old_value=old_value,
                new_value=value,
                user_id=user_id
            )
            self.change_history.append(change)
            
            # Run post-set hooks
            if param_key in self.post_set_hooks:
                for hook in self.post_set_hooks[param_key]:
                    try:
                        hook(module_id, param_id, old_value, value, user_id)
                    except Exception as e:
                        logger.error(f"Error in post-set hook for {module_id}.{param_id}: {e}")
            
            logger.info(f"Parameter {module_id}.{param_id} set to {value}")
            
            # Publish event if event bus is available
            if self.event_bus:
                self.event_bus.publish(
                    "config.parameter_changed",
                    {
                        "module_id": module_id,
                        "param_id": param_id,
                        "old_value": old_value,
                        "new_value": value,
                        "user_id": user_id
                    }
                )
            
            return True
    
    def load_preset(
        self,
        preset_id: str,
        user_id: str = None
    ) -> bool:
        """Load a parameter preset
        
        Args:
            preset_id: ID of the preset
            user_id: ID of user making the change
            
        Returns:
            bool: True if preset was loaded, False otherwise
        """
        with self.lock:
            # Check if preset exists
            if preset_id not in self.parameter_presets:
                logger.warning(f"Parameter preset {preset_id} not found")
                return False
            
            # Get preset
            preset = self.parameter_presets[preset_id]
            
            # Apply preset parameters
            success = True
            for module_id, params in preset.parameters.items():
                for param_id, value in params.items():
                    if not self.set_parameter(module_id, param_id, value, user_id):
                        logger.warning(f"Failed to set {module_id}.{param_id} from preset {preset_id}")
                        success = False
            
            logger.info(f"Parameter preset {preset_id} loaded")
            
            # Publish event if event bus is available
            if self.event_bus:
                self.event_bus.publish(
                    "config.preset_loaded",
                    {
                        "preset_id": preset_id,
                        "user_id": user_id
                    }
                )
            
            return success
    
    def save_preset(
        self,
        preset_id: str,
        name: str = None,
        description: str = None,
        module_ids: List[str] = None,
        param_ids: List[str] = None,
        tags: List[str] = None
    ) -> bool:
        """Save current parameters as a preset
        
        Args:
            preset_id: ID of the preset
            name: Display name of the preset
            description: Description of the preset
            module_ids: List of module IDs to include (None for all)
            param_ids: List of parameter IDs to include (None for all)
            tags: List of tags for categorizing the preset
            
        Returns:
            bool: True if preset was saved, False otherwise
        """
        with self.lock:
            # Collect parameters for preset
            parameters = {}
            for (mod_id, par_id), value in self.parameter_values.items():
                # Filter by module_ids if specified
                if module_ids is not None and mod_id not in module_ids:
                    continue
                
                # Filter by param_ids if specified
                if param_ids is not None and par_id not in param_ids:
                    continue
                
                # Add parameter to preset
                if mod_id not in parameters:
                    parameters[mod_id] = {}
                parameters[mod_id][par_id] = value
            
            # Register preset
            return self.register_preset(
                preset_id=preset_id,
                parameters=parameters,
                name=name,
                description=description,
                tags=tags
            )
    
    def get_parameter_definition(
        self,
        module_id: str,
        param_id: str
    ) -> Optional[ParameterDefinition]:
        """Get parameter definition
        
        Args:
            module_id: ID of the module that owns this parameter
            param_id: ID of the parameter
            
        Returns:
            ParameterDefinition: Parameter definition or None if not found
        """
        with self.lock:
            param_key = (module_id, param_id)
            return self.parameter_definitions.get(param_key)
    
    def get_parameters_by_module(
        self,
        module_id: str
    ) -> Dict[str, Any]:
        """Get all parameters for a module
        
        Args:
            module_id: ID of the module
            
        Returns:
            dict: Dictionary of parameter values by param_id
        """
        with self.lock:
            result = {}
            for (mod_id, par_id), value in self.parameter_values.items():
                if mod_id == module_id:
                    result[par_id] = value
            return result
    
    def get_parameters_by_group(
        self,
        group_id: str
    ) -> Dict[str, Dict[str, Any]]:
        """Get all parameters in a group
        
        Args:
            group_id: ID of the group
            
        Returns:
            dict: Dictionary of parameter values by module_id and param_id
        """
        with self.lock:
            result = {}
            for (mod_id, par_id), param_def in self.parameter_definitions.items():
                if param_def.group == group_id:
                    if mod_id not in result:
                        result[mod_id] = {}
                    result[mod_id][par_id] = self.get_parameter(mod_id, par_id)
            return result
    
    def get_parameters_by_tag(
        self,
        tag: str
    ) -> Dict[str, Dict[str, Any]]:
        """Get all parameters with a specific tag
        
        Args:
            tag: Tag to filter by
            
        Returns:
            dict: Dictionary of parameter values by module_id and param_id
        """
        with self.lock:
            result = {}
            for (mod_id, par_id), param_def in self.parameter_definitions.items():
                if tag in param_def.tags:
                    if mod_id not in result:
                        result[mod_id] = {}
                    result[mod_id][par_id] = self.get_parameter(mod_id, par_id)
            return result
    
    def get_all_parameters(self) -> Dict[str, Dict[str, Any]]:
        """Get all parameters
        
        Returns:
            dict: Dictionary of parameter values by module_id and param_id
        """
        with self.lock:
            result = {}
            for (mod_id, par_id) in self.parameter_definitions.keys():
                if mod_id not in result:
                    result[mod_id] = {}
                result[mod_id][par_id] = self.get_parameter(mod_id, par_id)
            return result
    
    def get_parameter_group(
        self,
        group_id: str
    ) -> Optional[ParameterGroup]:
        """Get parameter group
        
        Args:
            group_id: ID of the group
            
        Returns:
            ParameterGroup: Parameter group or None if not found
        """
        with self.lock:
            return self.parameter_groups.get(group_id)
    
    def get_all_groups(self) -> Dict[str, ParameterGroup]:
        """Get all parameter groups
        
        Returns:
            dict: Dictionary of parameter groups by group_id
        """
        with self.lock:
            return dict(self.parameter_groups)
    
    def get_preset(
        self,
        preset_id: str
    ) -> Optional[ParameterPreset]:
        """Get parameter preset
        
        Args:
            preset_id: ID of the preset
            
        Returns:
            ParameterPreset: Parameter preset or None if not found
        """
        with self.lock:
            return self.parameter_presets.get(preset_id)
    
    def get_presets_by_tag(
        self,
        tag: str
    ) -> Dict[str, ParameterPreset]:
        """Get all presets with a specific tag
        
        Args:
            tag: Tag to filter by
            
        Returns:
            dict: Dictionary of parameter presets by preset_id
        """
        with self.lock:
            result = {}
            for preset_id, preset in self.parameter_presets.items():
                if tag in preset.tags:
                    result[preset_id] = preset
            return result
    
    def get_all_presets(self) -> Dict[str, ParameterPreset]:
        """Get all parameter presets
        
        Returns:
            dict: Dictionary of parameter presets by preset_id
        """
        with self.lock:
            return dict(self.parameter_presets)
    
    def get_change_history(
        self,
        module_id: str = None,
        param_id: str = None,
        limit: int = None
    ) -> List[ParameterChange]:
        """Get parameter change history
        
        Args:
            module_id: Filter by module ID (optional)
            param_id: Filter by parameter ID (optional)
            limit: Maximum number of changes to return (optional)
            
        Returns:
            list: List of parameter changes
        """
        with self.lock:
            # Filter changes
            changes = self.change_history
            if module_id is not None:
                changes = [c for c in changes if c.module_id == module_id]
            if param_id is not None:
                changes = [c for c in changes if c.param_id == param_id]
            
            # Sort by timestamp (newest first)
            changes = sorted(changes, key=lambda c: c.timestamp, reverse=True)
            
            # Apply limit
            if limit is not None:
                changes = changes[:limit]
            
            return changes
    
    def register_validator(
        self,
        module_id: str,
        param_id: str,
        validation_func: Callable[[Any], bool]
    ) -> bool:
        """Register a custom validator for a parameter
        
        Args:
            module_id: ID of the module that owns this parameter
            param_id: ID of the parameter
            validation_func: Function to validate parameter values
            
        Returns:
            bool: True if validator was registered, False otherwise
        """
        with self.lock:
            param_key = (module_id, param_id)
            
            # Check if parameter exists
            if param_key not in self.parameter_definitions:
                logger.warning(f"Parameter {module_id}.{param_id} not registered")
                return False
            
            # Store validator
            self.validators[param_key] = validation_func
            
            # Update parameter definition
            self.parameter_definitions[param_key].validation_func = validation_func
            
            logger.info(f"Validator registered for {module_id}.{param_id}")
            return True
    
    def register_pre_set_hook(
        self,
        module_id: str,
        param_id: str,
        hook_func: Callable[[str, str, Any, Any, str], Any]
    ) -> bool:
        """Register a hook to run before setting a parameter
        
        The hook function receives:
        - module_id: ID of the module that owns this parameter
        - param_id: ID of the parameter
        - old_value: Current parameter value
        - new_value: New parameter value
        - user_id: ID of user making the change
        
        The hook function can:
        - Return None to allow the change
        - Return False to reject the change
        - Return a modified value to use instead
        
        Args:
            module_id: ID of the module that owns this parameter
            param_id: ID of the parameter
            hook_func: Function to run before setting parameter
            
        Returns:
            bool: True if hook was registered, False otherwise
        """
        with self.lock:
            param_key = (module_id, param_id)
            
            # Check if parameter exists
            if param_key not in self.parameter_definitions:
                logger.warning(f"Parameter {module_id}.{param_id} not registered")
                return False
            
            # Initialize hook list if needed
            if param_key not in self.pre_set_hooks:
                self.pre_set_hooks[param_key] = []
            
            # Add hook
            self.pre_set_hooks[param_key].append(hook_func)
            
            logger.info(f"Pre-set hook registered for {module_id}.{param_id}")
            return True
    
    def register_post_set_hook(
        self,
        module_id: str,
        param_id: str,
        hook_func: Callable[[str, str, Any, Any, str], None]
    ) -> bool:
        """Register a hook to run after setting a parameter
        
        The hook function receives:
        - module_id: ID of the module that owns this parameter
        - param_id: ID of the parameter
        - old_value: Previous parameter value
        - new_value: New parameter value
        - user_id: ID of user making the change
        
        Args:
            module_id: ID of the module that owns this parameter
            param_id: ID of the parameter
            hook_func: Function to run after setting parameter
            
        Returns:
            bool: True if hook was registered, False otherwise
        """
        with self.lock:
            param_key = (module_id, param_id)
            
            # Check if parameter exists
            if param_key not in self.parameter_definitions:
                logger.warning(f"Parameter {module_id}.{param_id} not registered")
                return False
            
            # Initialize hook list if needed
            if param_key not in self.post_set_hooks:
                self.post_set_hooks[param_key] = []
            
            # Add hook
            self.post_set_hooks[param_key].append(hook_func)
            
            logger.info(f"Post-set hook registered for {module_id}.{param_id}")
            return True
    
    def register_pre_load_hook(
        self,
        hook_func: Callable[[], None]
    ) -> bool:
        """Register a hook to run before loading configuration
        
        Args:
            hook_func: Function to run before loading configuration
            
        Returns:
            bool: True if hook was registered
        """
        with self.lock:
            self.pre_load_hooks.append(hook_func)
            logger.info("Pre-load hook registered")
            return True
    
    def register_post_load_hook(
        self,
        hook_func: Callable[[Dict[str, Dict[str, Any]]], None]
    ) -> bool:
        """Register a hook to run after loading configuration
        
        The hook function receives:
        - parameters: Dictionary of loaded parameters by module_id and param_id
        
        Args:
            hook_func: Function to run after loading configuration
            
        Returns:
            bool: True if hook was registered
        """
        with self.lock:
            self.post_load_hooks.append(hook_func)
            logger.info("Post-load hook registered")
            return True
    
    def register_pre_save_hook(
        self,
        hook_func: Callable[[], None]
    ) -> bool:
        """Register a hook to run before saving configuration
        
        Args:
            hook_func: Function to run before saving configuration
            
        Returns:
            bool: True if hook was registered
        """
        with self.lock:
            self.pre_save_hooks.append(hook_func)
            logger.info("Pre-save hook registered")
            return True
    
    def register_post_save_hook(
        self,
        hook_func: Callable[[str], None]
    ) -> bool:
        """Register a hook to run after saving configuration
        
        The hook function receives:
        - config_file: Path to the saved configuration file
        
        Args:
            hook_func: Function to run after saving configuration
            
        Returns:
            bool: True if hook was registered
        """
        with self.lock:
            self.post_save_hooks.append(hook_func)
            logger.info("Post-save hook registered")
            return True
    
    def load_config(
        self,
        config_file: str = None
    ) -> bool:
        """Load configuration from file
        
        Args:
            config_file: Path to configuration file (optional)
            
        Returns:
            bool: True if configuration was loaded, False otherwise
        """
        with self.lock:
            # Determine config file path
            if config_file is None:
                config_file = os.path.join(self.config_dir, "config.yaml")
            
            # Run pre-load hooks
            for hook in self.pre_load_hooks:
                try:
                    hook()
                except Exception as e:
                    logger.error(f"Error in pre-load hook: {e}")
            
            # Check if config file exists
            if not os.path.exists(config_file):
                logger.warning(f"Configuration file {config_file} not found")
                return False
            
            try:
                # Load configuration from file
                with open(config_file, "r") as f:
                    if config_file.endswith(".json"):
                        config_data = json.load(f)
                    else:
                        config_data = yaml.safe_load(f)
                
                # Process configuration data
                if not isinstance(config_data, dict):
                    logger.error(f"Invalid configuration format in {config_file}")
                    return False
                
                # Load parameters
                for module_id, params in config_data.get("parameters", {}).items():
                    for param_id, value in params.items():
                        param_key = (module_id, param_id)
                        if param_key in self.parameter_definitions:
                            self.parameter_values[param_key] = value
                
                # Load parameter groups
                for group_id, group_data in config_data.get("groups", {}).items():
                    self.register_parameter_group(
                        group_id=group_id,
                        name=group_data.get("name"),
                        description=group_data.get("description"),
                        parent_group=group_data.get("parent_group")
                    )
                
                # Load parameter presets
                for preset_id, preset_data in config_data.get("presets", {}).items():
                    self.register_preset(
                        preset_id=preset_id,
                        parameters=preset_data.get("parameters", {}),
                        name=preset_data.get("name"),
                        description=preset_data.get("description"),
                        tags=preset_data.get("tags")
                    )
                
                logger.info(f"Configuration loaded from {config_file}")
                
                # Run post-load hooks
                loaded_params = self.get_all_parameters()
                for hook in self.post_load_hooks:
                    try:
                        hook(loaded_params)
                    except Exception as e:
                        logger.error(f"Error in post-load hook: {e}")
                
                # Publish event if event bus is available
                if self.event_bus:
                    self.event_bus.publish(
                        "config.loaded",
                        {
                            "config_file": config_file
                        }
                    )
                
                return True
            
            except Exception as e:
                logger.error(f"Error loading configuration from {config_file}: {e}")
                return False
    
    def save_config(
        self,
        config_file: str = None,
        include_defaults: bool = False
    ) -> bool:
        """Save configuration to file
        
        Args:
            config_file: Path to configuration file (optional)
            include_defaults: Whether to include default values
            
        Returns:
            bool: True if configuration was saved, False otherwise
        """
        with self.lock:
            # Determine config file path
            if config_file is None:
                config_file = os.path.join(self.config_dir, "config.yaml")
            
            # Run pre-save hooks
            for hook in self.pre_save_hooks:
                try:
                    hook()
                except Exception as e:
                    logger.error(f"Error in pre-save hook: {e}")
            
            try:
                # Create configuration data
                config_data = {
                    "parameters": {},
                    "groups": {},
                    "presets": {}
                }
                
                # Add parameters
                for (mod_id, par_id), value in self.parameter_values.items():
                    # Skip parameters with default values if not including defaults
                    if not include_defaults:
                        param_def = self.parameter_definitions.get((mod_id, par_id))
                        if param_def and value == param_def.default_value:
                            continue
                    
                    # Add parameter to config data
                    if mod_id not in config_data["parameters"]:
                        config_data["parameters"][mod_id] = {}
                    config_data["parameters"][mod_id][par_id] = value
                
                # Add parameter groups
                for group_id, group in self.parameter_groups.items():
                    config_data["groups"][group_id] = group.to_dict()
                
                # Add parameter presets
                for preset_id, preset in self.parameter_presets.items():
                    config_data["presets"][preset_id] = preset.to_dict()
                
                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(config_file), exist_ok=True)
                
                # Save configuration to file
                with open(config_file, "w") as f:
                    if config_file.endswith(".json"):
                        json.dump(config_data, f, indent=2)
                    else:
                        yaml.dump(config_data, f, default_flow_style=False)
                
                logger.info(f"Configuration saved to {config_file}")
                
                # Run post-save hooks
                for hook in self.post_save_hooks:
                    try:
                        hook(config_file)
                    except Exception as e:
                        logger.error(f"Error in post-save hook: {e}")
                
                # Publish event if event bus is available
                if self.event_bus:
                    self.event_bus.publish(
                        "config.saved",
                        {
                            "config_file": config_file
                        }
                    )
                
                return True
            
            except Exception as e:
                logger.error(f"Error saving configuration to {config_file}: {e}")
                return False
    
    def export_config(
        self,
        format_type: str = "yaml"
    ) -> str:
        """Export configuration as string
        
        Args:
            format_type: Format type ("yaml" or "json")
            
        Returns:
            str: Configuration as string
        """
        with self.lock:
            # Create configuration data
            config_data = {
                "parameters": {},
                "groups": {},
                "presets": {}
            }
            
            # Add parameters
            for (mod_id, par_id), value in self.parameter_values.items():
                param_def = self.parameter_definitions.get((mod_id, par_id))
                if param_def and param_def.secret:
                    # Mask secret values
                    value = "********"
                
                # Add parameter to config data
                if mod_id not in config_data["parameters"]:
                    config_data["parameters"][mod_id] = {}
                config_data["parameters"][mod_id][par_id] = value
            
            # Add parameter groups
            for group_id, group in self.parameter_groups.items():
                config_data["groups"][group_id] = group.to_dict()
            
            # Add parameter presets
            for preset_id, preset in self.parameter_presets.items():
                config_data["presets"][preset_id] = preset.to_dict()
            
            # Export configuration as string
            if format_type.lower() == "json":
                return json.dumps(config_data, indent=2)
            else:
                return yaml.dump(config_data, default_flow_style=False)
    
    def import_config(
        self,
        config_data: Union[str, Dict],
        format_type: str = None,
        user_id: str = None
    ) -> bool:
        """Import configuration from string or dictionary
        
        Args:
            config_data: Configuration as string or dictionary
            format_type: Format type ("yaml" or "json") if string
            user_id: ID of user making the change
            
        Returns:
            bool: True if configuration was imported, False otherwise
        """
        with self.lock:
            try:
                # Parse configuration data if string
                if isinstance(config_data, str):
                    if format_type is None:
                        format_type = "yaml"
                    
                    if format_type.lower() == "json":
                        config_data = json.loads(config_data)
                    else:
                        config_data = yaml.safe_load(config_data)
                
                # Process configuration data
                if not isinstance(config_data, dict):
                    logger.error("Invalid configuration format")
                    return False
                
                # Import parameters
                for module_id, params in config_data.get("parameters", {}).items():
                    for param_id, value in params.items():
                        # Skip masked secret values
                        if value == "********":
                            continue
                        
                        # Set parameter value
                        self.set_parameter(module_id, param_id, value, user_id)
                
                # Import parameter groups
                for group_id, group_data in config_data.get("groups", {}).items():
                    self.register_parameter_group(
                        group_id=group_id,
                        name=group_data.get("name"),
                        description=group_data.get("description"),
                        parent_group=group_data.get("parent_group")
                    )
                
                # Import parameter presets
                for preset_id, preset_data in config_data.get("presets", {}).items():
                    self.register_preset(
                        preset_id=preset_id,
                        parameters=preset_data.get("parameters", {}),
                        name=preset_data.get("name"),
                        description=preset_data.get("description"),
                        tags=preset_data.get("tags")
                    )
                
                logger.info("Configuration imported")
                
                # Publish event if event bus is available
                if self.event_bus:
                    self.event_bus.publish(
                        "config.imported",
                        {
                            "user_id": user_id
                        }
                    )
                
                return True
            
            except Exception as e:
                logger.error(f"Error importing configuration: {e}")
                return False
    
    def reset_to_defaults(
        self,
        module_id: str = None,
        param_id: str = None,
        user_id: str = None
    ) -> bool:
        """Reset parameters to default values
        
        Args:
            module_id: Reset only parameters for this module (optional)
            param_id: Reset only this parameter (requires module_id)
            user_id: ID of user making the change
            
        Returns:
            bool: True if parameters were reset, False otherwise
        """
        with self.lock:
            success = True
            
            # Reset specific parameter
            if module_id is not None and param_id is not None:
                param_key = (module_id, param_id)
                if param_key in self.parameter_definitions:
                    default_value = self.parameter_definitions[param_key].default_value
                    if not self.set_parameter(module_id, param_id, default_value, user_id):
                        success = False
                else:
                    logger.warning(f"Parameter {module_id}.{param_id} not registered")
                    success = False
            
            # Reset all parameters for a module
            elif module_id is not None:
                for (mod_id, par_id), param_def in self.parameter_definitions.items():
                    if mod_id == module_id:
                        if not self.set_parameter(mod_id, par_id, param_def.default_value, user_id):
                            success = False
            
            # Reset all parameters
            else:
                for (mod_id, par_id), param_def in self.parameter_definitions.items():
                    if not self.set_parameter(mod_id, par_id, param_def.default_value, user_id):
                        success = False
            
            logger.info(f"Parameters reset to defaults")
            
            # Publish event if event bus is available
            if self.event_bus:
                self.event_bus.publish(
                    "config.reset_to_defaults",
                    {
                        "module_id": module_id,
                        "param_id": param_id,
                        "user_id": user_id
                    }
                )
            
            return success


# Example usage
if __name__ == "__main__":
    # Create config registry
    config_registry = ConfigRegistry()
    
    # Register parameters
    config_registry.register_parameter(
        module_id="system",
        param_id="log_level",
        default_value="INFO",
        options=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        description="Logging level",
        group="logging"
    )
    
    config_registry.register_parameter(
        module_id="system",
        param_id="max_threads",
        default_value=4,
        param_type=int,
        min_value=1,
        max_value=16,
        description="Maximum number of threads",
        group="performance"
    )
    
    config_registry.register_parameter(
        module_id="telegram",
        param_id="bot_token",
        default_value="",
        description="Telegram bot token",
        group="api_keys",
        secret=True
    )
    
    # Register parameter groups
    config_registry.register_parameter_group(
        group_id="logging",
        name="Logging",
        description="Logging configuration"
    )
    
    config_registry.register_parameter_group(
        group_id="performance",
        name="Performance",
        description="Performance configuration"
    )
    
    config_registry.register_parameter_group(
        group_id="api_keys",
        name="API Keys",
        description="API keys configuration"
    )
    
    # Register parameter preset
    config_registry.register_preset(
        preset_id="high_performance",
        name="High Performance",
        description="Configuration for high performance",
        parameters={
            "system": {
                "max_threads": 8,
                "log_level": "WARNING"
            }
        }
    )
    
    # Set parameter values
    config_registry.set_parameter("system", "log_level", "DEBUG")
    config_registry.set_parameter("system", "max_threads", 8)
    config_registry.set_parameter("telegram", "bot_token", "1234567890:ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    
    # Get parameter values
    print(f"Log level: {config_registry.get_parameter('system', 'log_level')}")
    print(f"Max threads: {config_registry.get_parameter('system', 'max_threads')}")
    print(f"Bot token: {config_registry.get_parameter('telegram', 'bot_token')}")
    
    # Get parameters by group
    print(f"Logging parameters: {config_registry.get_parameters_by_group('logging')}")
    print(f"Performance parameters: {config_registry.get_parameters_by_group('performance')}")
    
    # Load preset
    config_registry.load_preset("high_performance")
    print(f"Log level after preset: {config_registry.get_parameter('system', 'log_level')}")
    print(f"Max threads after preset: {config_registry.get_parameter('system', 'max_threads')}")
    
    # Save configuration
    config_registry.save_config()
    
    # Export configuration
    print(f"Configuration:\n{config_registry.export_config()}")
