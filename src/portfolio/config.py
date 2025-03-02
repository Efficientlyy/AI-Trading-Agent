"""
Portfolio configuration utilities.

This module provides functions for loading and validating portfolio
configuration settings from YAML files.
"""

import os
from decimal import Decimal
from pathlib import Path
from typing import Dict, Any, Optional, List

import yaml

from src.portfolio.portfolio_manager import RiskParameters


def get_project_root() -> str:
    """Get the absolute path to the project root directory.
    
    Returns:
        Absolute path to the project root directory
    """
    # The file is in src/portfolio/config.py, so we need to go up two levels
    current_file = os.path.abspath(__file__)
    src_dir = os.path.dirname(os.path.dirname(current_file))
    return os.path.dirname(src_dir)


def load_portfolio_config(portfolio_name: str = "main") -> Dict[str, Any]:
    """Load portfolio configuration from YAML file.
    
    Args:
        portfolio_name: Name of the portfolio configuration to load
        
    Returns:
        Dictionary containing portfolio configuration
        
    Raises:
        FileNotFoundError: If the portfolio configuration file is not found
        ValueError: If the specified portfolio is not defined in the configuration
    """
    config_path = os.path.join(get_project_root(), "config", "portfolio.yaml")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Portfolio configuration file not found: {config_path}")
    
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    
    # Look for the specified portfolio
    portfolio_key = f"{portfolio_name}_portfolio"
    if portfolio_key not in config:
        # Try looking for the portfolio by name
        for key, value in config.items():
            if key.endswith("_portfolio") and isinstance(value, dict) and value.get("name") == portfolio_name:
                portfolio_key = key
                break
        else:
            raise ValueError(f"Portfolio '{portfolio_name}' not found in configuration")
    
    portfolio_config = config[portfolio_key]
    
    # Add common settings
    if "performance_tracking" in config:
        portfolio_config["performance_tracking"] = config["performance_tracking"]
    
    if "logging" in config:
        portfolio_config["logging"] = config["logging"]
    
    return portfolio_config


def create_risk_parameters_from_config(config: Dict[str, Any]) -> RiskParameters:
    """Create RiskParameters from configuration.
    
    Args:
        config: Portfolio configuration dictionary
        
    Returns:
        RiskParameters object
    """
    risk_config = config.get("risk_management", {})
    
    return RiskParameters(
        max_position_size=Decimal(str(risk_config.get("max_position_size", 0.1))),
        max_risk_per_trade_pct=Decimal(str(risk_config.get("max_risk_per_trade_pct", 0.01))),
        max_risk_per_day_pct=Decimal(str(risk_config.get("max_risk_per_day_pct", 0.05))),
        max_open_positions=int(risk_config.get("max_open_positions", 5)),
        max_open_positions_per_symbol=int(risk_config.get("max_open_positions_per_symbol", 2)),
        max_drawdown_pct=Decimal(str(risk_config.get("max_drawdown_pct", 0.2)))
    )


def get_default_exit_levels(
    config: Dict[str, Any], 
    price: Decimal, 
    direction: str
) -> Dict[str, Decimal]:
    """Get default take profit and stop loss levels based on configuration.
    
    Args:
        config: Portfolio configuration dictionary
        price: Current price
        direction: Trade direction ("long" or "short")
        
    Returns:
        Dictionary containing take_profit and stop_loss prices
    """
    take_profit_pct = Decimal(str(config.get("default_take_profit_pct", 0.02)))
    stop_loss_pct = Decimal(str(config.get("default_stop_loss_pct", 0.01)))
    
    if direction == "long":
        take_profit = price * (Decimal("1") + take_profit_pct)
        stop_loss = price * (Decimal("1") - stop_loss_pct)
    else:  # short
        take_profit = price * (Decimal("1") - take_profit_pct)
        stop_loss = price * (Decimal("1") + stop_loss_pct)
    
    return {
        "take_profit_price": take_profit,
        "stop_loss_price": stop_loss
    }


def calculate_position_size(
    config: Dict[str, Any],
    portfolio_value: Decimal,
    price: Decimal,
    stop_loss: Optional[Decimal] = None,
    risk_multiplier: Decimal = Decimal("1.0")
) -> Decimal:
    """Calculate position size based on configuration and risk parameters.
    
    Args:
        config: Portfolio configuration dictionary
        portfolio_value: Current portfolio value
        price: Current price
        stop_loss: Optional stop loss price
        risk_multiplier: Multiplier for risk calculation (1.0 = normal risk)
        
    Returns:
        Position size in base currency units
    """
    position_sizing = config.get("position_sizing", {})
    mode = position_sizing.get("mode", "risk_based")
    
    if mode == "fixed":
        # Fixed position size
        return Decimal(str(position_sizing.get("fixed_size", 0.1)))
    
    elif mode == "portfolio_percentage":
        # Percentage of portfolio
        portfolio_percentage = Decimal(str(position_sizing.get("portfolio_percentage", 0.05)))
        return portfolio_value * portfolio_percentage / price
    
    elif mode == "risk_based":
        # Risk-based position sizing
        risk_config = config.get("risk_management", {})
        risk_pct = Decimal(str(risk_config.get("max_risk_per_trade_pct", 0.01)))
        
        # Calculate the risk amount
        risk_amount = portfolio_value * risk_pct * risk_multiplier
        
        if stop_loss is not None and price != stop_loss:
            # Calculate position size based on stop loss
            risk_per_unit = abs(price - stop_loss)
            return risk_amount / risk_per_unit
        else:
            # Default to a 1% risk if no stop loss is provided
            default_risk = price * Decimal("0.01")
            return risk_amount / default_risk
    
    else:
        # Default to a small fixed size
        return Decimal("0.01")


def validate_portfolio_config(config: Dict[str, Any]) -> List[str]:
    """Validate portfolio configuration and return any errors.
    
    Args:
        config: Portfolio configuration dictionary
        
    Returns:
        List of error messages, empty if validation succeeded
    """
    errors = []
    
    # Check required fields
    required_fields = ["name", "initial_balance", "base_currency"]
    for field in required_fields:
        if field not in config:
            errors.append(f"Missing required field: {field}")
    
    # Validate risk management parameters
    if "risk_management" in config:
        risk_config = config["risk_management"]
        
        # Check risk parameters are within reasonable ranges
        if "max_position_size" in risk_config and (
            risk_config["max_position_size"] <= 0 or risk_config["max_position_size"] > 1
        ):
            errors.append("max_position_size must be between 0 and 1")
        
        if "max_risk_per_trade_pct" in risk_config and (
            risk_config["max_risk_per_trade_pct"] <= 0 or risk_config["max_risk_per_trade_pct"] > 0.1
        ):
            errors.append("max_risk_per_trade_pct must be between 0 and 0.1 (10%)")
        
        if "max_risk_per_day_pct" in risk_config and (
            risk_config["max_risk_per_day_pct"] <= 0 or risk_config["max_risk_per_day_pct"] > 0.3
        ):
            errors.append("max_risk_per_day_pct must be between 0 and 0.3 (30%)")
    
    # Validate position sizing
    if "position_sizing" in config:
        sizing_config = config["position_sizing"]
        
        if "mode" in sizing_config and sizing_config["mode"] not in ["fixed", "risk_based", "portfolio_percentage"]:
            errors.append("position_sizing mode must be one of: fixed, risk_based, portfolio_percentage")
    
    return errors 