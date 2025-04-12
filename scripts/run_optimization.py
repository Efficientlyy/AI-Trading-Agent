#!/usr/bin/env python
"""
Optimization Runner Script

This script provides a command-line interface for running genetic algorithm
optimization experiments on trading strategies under various market conditions.
"""

import os
import sys
import argparse
import json
import yaml
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from ai_trading_agent.optimization.genetic_algorithm import (
    GeneticAlgorithm, Parameter, ParameterType, create_parameter_space,
    SelectionMethod, CrossoverMethod, MutationMethod
)
from ai_trading_agent.optimization.strategy_comparison import (
    StrategyComparison, compare_strategies
)
from ai_trading_agent.optimization.market_simulator import (
    MarketSimulator, MarketCondition, generate_market_scenarios
)
from ai_trading_agent.agent.factory import create_agent_from_config, is_rust_available
from ai_trading_agent.common.logging_config import logger, setup_logging

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run optimization experiments")
    
    # Main operation mode
    parser.add_argument(
        "--mode",
        choices=["optimize", "compare", "simulate"],
        default="optimize",
        help="Operation mode: optimize a strategy, compare strategies, or simulate market conditions"
    )
    
    # Configuration
    parser.add_argument(
        "--config",
        type=str,
        default="config/agent_config_template.yaml",
        help="Path to the agent configuration file"
    )
    
    parser.add_argument(
        "--param-space",
        type=str,
        help="Path to parameter space definition file (JSON)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="optimization_results",
        help="Directory for saving results"
    )
    
    # Genetic algorithm parameters
    parser.add_argument(
        "--population-size",
        type=int,
        default=50,
        help="Population size for genetic algorithm"
    )
    
    parser.add_argument(
        "--generations",
        type=int,
        default=30,
        help="Number of generations for genetic algorithm"
    )
    
    parser.add_argument(
        "--selection-method",
        choices=["tournament", "roulette", "rank"],
        default="tournament",
        help="Selection method for genetic algorithm"
    )
    
    parser.add_argument(
        "--crossover-method",
        choices=["uniform", "single_point", "two_point"],
        default="uniform",
        help="Crossover method for genetic algorithm"
    )
    
    parser.add_argument(
        "--mutation-method",
        choices=["uniform", "gaussian", "adaptive"],
        default="uniform",
        help="Mutation method for genetic algorithm"
    )
    
    # Market simulation parameters
    parser.add_argument(
        "--market-condition",
        choices=[c.value for c in MarketCondition],
        default="bull_market",
        help="Market condition to simulate"
    )
    
    parser.add_argument(
        "--start-date",
        type=str,
        default="2020-01-01",
        help="Start date for simulation"
    )
    
    parser.add_argument(
        "--end-date",
        type=str,
        default="2020-12-31",
        help="End date for simulation"
    )
    
    # Comparison parameters
    parser.add_argument(
        "--strategies",
        type=str,
        help="Path to strategies definition file (JSON)"
    )
    
    # Performance
    parser.add_argument(
        "--use-rust",
        action="store_true",
        help="Use Rust-accelerated components if available"
    )
    
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Use parallel processing"
    )
    
    parser.add_argument(
        "--max-workers",
        type=int,
        help="Maximum number of parallel workers"
    )
    
    # Misc
    parser.add_argument(
        "--random-seed",
        type=int,
        help="Random seed for reproducibility"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    return parser.parse_args()

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from a YAML file."""
    if not os.path.isabs(config_path):
        config_path = os.path.join(project_root, config_path)
    
    logger.info(f"Loading configuration from {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    return config

def load_param_space(param_space_path: str) -> Dict[str, Any]:
    """Load parameter space from a JSON file."""
    if not os.path.isabs(param_space_path):
        param_space_path = os.path.join(project_root, param_space_path)
    
    logger.info(f"Loading parameter space from {param_space_path}")
    with open(param_space_path, "r") as f:
        param_space = json.load(f)
    
    return param_space

def load_strategies(strategies_path: str) -> Dict[str, Dict[str, Any]]:
    """Load strategies from a JSON file."""
    if not os.path.isabs(strategies_path):
        strategies_path = os.path.join(project_root, strategies_path)
    
    logger.info(f"Loading strategies from {strategies_path}")
    with open(strategies_path, "r") as f:
        strategies = json.load(f)
    
    return strategies

def run_backtest_with_params(params: Dict[str, Any], config: Dict[str, Any], use_rust: bool = False) -> float:
    """
    Run a backtest with the given strategy parameters and return a fitness score.
    
    Args:
        params: Dictionary of strategy parameters
        config: Base configuration
        use_rust: Whether to use Rust-accelerated components
        
    Returns:
        Fitness score (higher is better)
    """
    # Create a copy of the configuration
    config_copy = config.copy()
    
    # Update strategy parameters
    if "strategy" not in config_copy:
        config_copy["strategy"] = {}
    
    if "config" not in config_copy["strategy"]:
        config_copy["strategy"]["config"] = {}
    
    # Add parameters to strategy config
    for key, value in params.items():
        config_copy["strategy"]["config"][key] = value
    
    try:
        # Create and run agent
        agent = create_agent_from_config(config_copy, use_rust=use_rust)
        results = agent.run()
        
        if results and "performance_metrics" in results:
            metrics = results["performance_metrics"]
            
            # Use Sharpe ratio as fitness by default
            if "sharpe_ratio" in metrics:
                return metrics["sharpe_ratio"]
            
            # Fall back to total return
            if "total_return" in metrics:
                return metrics["total_return"]
            
            # Last resort: first metric
            return list(metrics.values())[0]
        
        return -1.0  # No results
    
    except Exception as e:
        logger.error(f"Error running backtest: {e}")
        return -1.0  # Error

def run_optimization(args):
    """Run optimization experiment."""
    # Load configuration
    config = load_config(args.config)
    
    # Load parameter space
    if args.param_space:
        param_dict = load_param_space(args.param_space)
    else:
        # Default parameter space for a simple moving average crossover strategy
        param_dict = {
            "fast_period": {
                "type": "discrete",
                "values": list(range(5, 30))
            },
            "slow_period": {
                "type": "discrete",
                "values": list(range(20, 100))
            },
            "threshold": {
                "type": "continuous",
                "min": 0.0,
                "max": 0.05,
                "step": 0.01
            }
        }
    
    # Create parameter space
    parameters = create_parameter_space(param_dict)
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = os.path.join(args.output_dir, f"optimization_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create fitness function
    def fitness_func(params):
        return run_backtest_with_params(params, config, args.use_rust)
    
    # Map string enum values to actual enums
    selection_method = SelectionMethod(args.selection_method)
    crossover_method = CrossoverMethod(args.crossover_method)
    mutation_method = MutationMethod(args.mutation_method)
    
    # Create genetic algorithm
    ga = GeneticAlgorithm(
        parameters=parameters,
        fitness_func=fitness_func,
        population_size=args.population_size,
        generations=args.generations,
        selection_method=selection_method,
        crossover_method=crossover_method,
        mutation_method=mutation_method,
        parallel=args.parallel,
        max_workers=args.max_workers,
        log_dir=output_dir,
        random_seed=args.random_seed
    )
    
    # Run optimization
    logger.info("Starting genetic algorithm optimization")
    best_individual, history_df = ga.evolve()
    
    # Print results
    logger.info("Optimization completed")
    logger.info(f"Best parameters: {best_individual.params}")
    logger.info(f"Best fitness: {best_individual.fitness}")
    
    if best_individual.metrics:
        logger.info("Performance metrics:")
        for key, value in best_individual.metrics.items():
            if isinstance(value, float):
                logger.info(f"  {key}: {value:.4f}")
            else:
                logger.info(f"  {key}: {value}")
    
    # Save best parameters to config file
    best_config = config.copy()
    if "strategy" not in best_config:
        best_config["strategy"] = {}
    
    if "config" not in best_config["strategy"]:
        best_config["strategy"]["config"] = {}
    
    # Add optimized parameters to strategy config
    for key, value in best_individual.params.items():
        best_config["strategy"]["config"][key] = value
    
    # Save optimized config
    optimized_config_path = os.path.join(output_dir, "optimized_config.yaml")
    with open(optimized_config_path, "w") as f:
        yaml.dump(best_config, f, default_flow_style=False)
    
    logger.info(f"Optimized configuration saved to {optimized_config_path}")
    
    return best_individual, history_df

def run_comparison(args):
    """Run strategy comparison experiment."""
    # Load configuration
    config = load_config(args.config)
    
    # Load strategies
    if args.strategies:
        strategies = load_strategies(args.strategies)
    else:
        # Default strategies for comparison
        strategies = {
            "MovingAverageCrossover": {
                "type": "MovingAverageCrossover",
                "config": {
                    "fast_period": 10,
                    "slow_period": 50,
                    "threshold": 0.0
                }
            },
            "RSI": {
                "type": "RSI",
                "config": {
                    "period": 14,
                    "overbought": 70,
                    "oversold": 30
                }
            },
            "SentimentStrategy": {
                "type": "SentimentStrategy",
                "config": {
                    "sentiment_threshold": 0.3,
                    "position_size_pct": 0.1
                }
            }
        }
    
    # Create market conditions
    market_conditions = {
        "bull_market": {
            "start_date": args.start_date,
            "end_date": args.end_date,
            "description": "Bull market"
        },
        "bear_market": {
            "start_date": args.start_date,
            "end_date": args.end_date,
            "description": "Bear market"
        },
        "volatile": {
            "start_date": args.start_date,
            "end_date": args.end_date,
            "description": "Volatile market"
        }
    }
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = os.path.join(args.output_dir, f"comparison_{timestamp}")
    
    # Run comparison
    logger.info("Starting strategy comparison")
    comparison_df = compare_strategies(
        base_config=config,
        strategies=strategies,
        market_conditions=market_conditions,
        output_dir=output_dir,
        use_rust=args.use_rust
    )
    
    # Print results
    logger.info("Comparison completed")
    logger.info(f"Results saved to {output_dir}")
    
    return comparison_df

def run_simulation(args):
    """Run market simulation experiment."""
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = os.path.join(args.output_dir, f"simulation_{timestamp}")
    
    # Create simulator
    simulator = MarketSimulator(
        start_date=args.start_date,
        end_date=args.end_date,
        random_seed=args.random_seed
    )
    
    # Generate market data
    logger.info(f"Generating {args.market_condition} market data")
    market_data = simulator.generate_market_condition(
        condition=MarketCondition(args.market_condition)
    )
    
    # Save and plot market data
    simulator.save_market_data(market_data, output_dir, args.market_condition)
    simulator.plot_market_data(market_data, output_dir, args.market_condition)
    
    logger.info(f"Market data saved to {output_dir}")
    
    return market_data

def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Set up logging
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(log_level=log_level)
    
    # Check if Rust is available
    if args.use_rust and not is_rust_available():
        logger.warning("Rust components requested but not available, falling back to Python")
        args.use_rust = False
    
    # Run selected mode
    if args.mode == "optimize":
        run_optimization(args)
    elif args.mode == "compare":
        run_comparison(args)
    elif args.mode == "simulate":
        run_simulation(args)
    else:
        logger.error(f"Unknown mode: {args.mode}")
        sys.exit(1)

if __name__ == "__main__":
    main()
