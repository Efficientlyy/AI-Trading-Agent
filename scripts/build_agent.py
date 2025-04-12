#!/usr/bin/env python
"""
Agent Builder Script

This script demonstrates how to build and run a trading agent using the factory system.
It loads configuration from a YAML file, validates it, creates the agent components,
and runs the agent.

Usage:
    python build_agent.py --config path/to/config.yaml [--use-rust]

Example:
    python build_agent.py --config config/my_agent_config.yaml --use-rust
"""
import os
import sys
import argparse
import yaml
from pathlib import Path

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from ai_trading_agent.common.logging_config import logger, setup_logging
from ai_trading_agent.common.config_validator import validate_agent_config, check_config_compatibility
from ai_trading_agent.agent.factory import (
    create_agent_from_config, 
    register_custom_component, 
    is_rust_available
)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Build and run a trading agent")
    parser.add_argument(
        "--config", 
        type=str, 
        default="config/agent_config_template.yaml",
        help="Path to the agent configuration file"
    )
    parser.add_argument(
        "--use-rust",
        action="store_true",
        help="Use Rust-accelerated components if available"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to save the results (CSV format)"
    )
    return parser.parse_args()

def load_config(config_path):
    """Load configuration from a YAML file."""
    logger.info(f"Loading configuration from {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def register_custom_components(config):
    """Register custom components from configuration."""
    if "custom_components" not in config:
        return
    
    custom_components = config["custom_components"]
    if not custom_components:
        return
    
    logger.info(f"Registering {len(custom_components)} custom components")
    
    for component in custom_components:
        component_type = component.get("type")
        name = component.get("name")
        module_path = component.get("module_path")
        class_name = component.get("class_name")
        
        if not all([component_type, name, module_path, class_name]):
            logger.warning(f"Skipping incomplete custom component definition: {component}")
            continue
        
        try:
            register_custom_component(component_type, name, module_path, class_name)
            logger.info(f"Registered custom component: {name} ({component_type})")
        except (ValueError, ImportError) as e:
            logger.error(f"Failed to register custom component {name}: {e}")

def save_results(results, output_path):
    """Save results to a file."""
    if not results:
        logger.warning("No results to save")
        return
    
    if "portfolio_history" not in results:
        logger.warning("No portfolio history in results")
        return
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save portfolio history
    portfolio_history = results["portfolio_history"]
    portfolio_history.to_csv(output_path)
    logger.info(f"Saved portfolio history to {output_path}")
    
    # Save performance metrics if available
    if "performance_metrics" in results:
        metrics_path = os.path.splitext(output_path)[0] + "_metrics.csv"
        metrics = results["performance_metrics"]
        
        # Convert metrics to DataFrame
        import pandas as pd
        metrics_df = pd.DataFrame([metrics])
        
        metrics_df.to_csv(metrics_path, index=False)
        logger.info(f"Saved performance metrics to {metrics_path}")

def main():
    """Main function to build and run the agent."""
    # Parse command line arguments
    args = parse_args()
    
    # Set up logging
    setup_logging()
    
    # Load configuration
    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(project_root, config_path)
    
    if not os.path.exists(config_path):
        logger.error(f"Configuration file not found: {config_path}")
        sys.exit(1)
    
    config = load_config(config_path)
    
    # Validate configuration
    is_valid, error_message = validate_agent_config(config)
    if not is_valid:
        logger.error(f"Invalid configuration: {error_message}")
        sys.exit(1)
    
    # Check for compatibility issues
    warnings = check_config_compatibility(config)
    for warning in warnings:
        logger.warning(warning)
    
    # Register custom components
    register_custom_components(config)
    
    # Determine whether to use Rust
    use_rust = args.use_rust or config.get("backtest", {}).get("use_rust", False)
    
    if use_rust and not is_rust_available():
        logger.warning("Rust components requested but not available, falling back to Python")
        use_rust = False
    
    # Create agent
    try:
        logger.info(f"Creating agent from configuration (use_rust={use_rust})")
        agent = create_agent_from_config(config, use_rust=use_rust)
        
        # Run agent
        logger.info("Running agent")
        results = agent.run()
        
        # Process results
        if results:
            logger.info("Agent run completed successfully")
            if "performance_metrics" in results:
                logger.info("Performance Metrics:")
                for key, value in results["performance_metrics"].items():
                    if isinstance(value, float):
                        logger.info(f"  {key}: {value:.4f}")
                    else:
                        logger.info(f"  {key}: {value}")
            
            # Save results if output path is specified
            if args.output:
                save_results(results, args.output)
        else:
            logger.warning("Agent run did not return results")
        
    except Exception as e:
        logger.error(f"Error creating or running agent: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
