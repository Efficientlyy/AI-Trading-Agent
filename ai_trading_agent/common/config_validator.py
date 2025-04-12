"""
Configuration validator for the AI Trading Agent.
Provides functionality to validate agent configuration against schemas.
"""
import json
from typing import Dict, Any, List, Optional, Union, Tuple
from jsonschema import validate, ValidationError
from .logging_config import logger
from .config_loader import get_config, get_config_value

# Define the schema for the agent configuration
AGENT_CONFIG_SCHEMA = {
    "type": "object",
    "properties": {
        "data_manager": {
            "type": "object",
            "required": ["type", "config"],
            "properties": {
                "type": {"type": "string"},
                "config": {
                    "type": "object",
                    "required": ["data_dir"],
                }
            }
        },
        "strategy": {
            "type": "object",
            "required": ["type", "config"],
            "properties": {
                "type": {"type": "string"},
                "config": {
                    "type": "object",
                    "required": ["name", "symbols"],
                }
            }
        },
        "risk_manager": {
            "type": "object",
            "required": ["type", "config"],
            "properties": {
                "type": {"type": "string"},
                "config": {"type": "object"}
            }
        },
        "portfolio_manager": {
            "type": "object",
            "required": ["type", "config"],
            "properties": {
                "type": {"type": "string"},
                "config": {
                    "type": "object",
                    "required": ["initial_capital"],
                }
            }
        },
        "execution_handler": {
            "type": "object",
            "required": ["type", "config"],
            "properties": {
                "type": {"type": "string"},
                "config": {"type": "object"}
            }
        },
        "backtest": {
            "type": "object",
            "required": ["start_date", "end_date", "symbols", "initial_capital"],
        }
    },
    "required": ["data_manager", "strategy", "risk_manager", "portfolio_manager", "execution_handler"]
}

def validate_agent_config(config: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """
    Validate the agent configuration against the schema.

    Args:
        config: The configuration to validate.

    Returns:
        Tuple of (is_valid, error_message). If is_valid is True, error_message is None.
    """
    try:
        validate(instance=config, schema=AGENT_CONFIG_SCHEMA)
        logger.info("Agent configuration validated successfully")
        return True, None
    except ValidationError as e:
        error_message = f"Agent configuration validation failed: {e.message}"
        logger.error(error_message)
        return False, error_message

def validate_component_config(component_type: str, component_config: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """
    Validate a specific component's configuration.

    Args:
        component_type: The type of component (e.g., 'data_manager', 'strategy').
        component_config: The component's configuration.

    Returns:
        Tuple of (is_valid, error_message). If is_valid is True, error_message is None.
    """
    if component_type not in AGENT_CONFIG_SCHEMA["properties"]:
        return False, f"Unknown component type: {component_type}"
    
    schema = AGENT_CONFIG_SCHEMA["properties"][component_type]
    try:
        validate(instance=component_config, schema=schema)
        logger.info(f"{component_type.capitalize()} configuration validated successfully")
        return True, None
    except ValidationError as e:
        error_message = f"{component_type.capitalize()} configuration validation failed: {e.message}"
        logger.error(error_message)
        return False, error_message

def check_config_compatibility(config: Dict[str, Any]) -> List[str]:
    """
    Check for compatibility issues between different components in the configuration.

    Args:
        config: The complete agent configuration.

    Returns:
        List of warning messages. Empty list if no warnings.
    """
    warnings = []
    
    # Check if strategy symbols match backtest symbols
    if "strategy" in config and "backtest" in config:
        strategy_symbols = config["strategy"]["config"].get("symbols", [])
        backtest_symbols = config["backtest"].get("symbols", [])
        
        if set(strategy_symbols) != set(backtest_symbols):
            warnings.append(
                f"Strategy symbols {strategy_symbols} do not match backtest symbols {backtest_symbols}"
            )
    
    # Check if initial_capital is consistent
    if "portfolio_manager" in config and "backtest" in config:
        pm_capital = config["portfolio_manager"]["config"].get("initial_capital")
        bt_capital = config["backtest"].get("initial_capital")
        
        if pm_capital is not None and bt_capital is not None and pm_capital != bt_capital:
            warnings.append(
                f"Portfolio manager initial capital ({pm_capital}) does not match backtest initial capital ({bt_capital})"
            )
    
    # Add more compatibility checks as needed
    
    return warnings

def validate_full_config() -> Tuple[bool, Optional[str], List[str]]:
    """
    Validate the full agent configuration from the config loader.

    Returns:
        Tuple of (is_valid, error_message, warnings).
        If is_valid is True, error_message is None.
    """
    config = get_config()
    is_valid, error_message = validate_agent_config(config)
    warnings = check_config_compatibility(config) if is_valid else []
    
    return is_valid, error_message, warnings
