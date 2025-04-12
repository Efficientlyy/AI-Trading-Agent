"""
Genetic Algorithm Optimizer for trading strategy parameter optimization.

This module provides a comprehensive genetic algorithm framework for optimizing
trading strategy parameters. It includes:
- Parameter space definition with various types (continuous, discrete, categorical)
- Advanced selection, crossover, and mutation operators
- Multi-objective optimization capabilities
- Parallel evaluation of fitness functions
- Detailed logging and visualization of optimization progress
"""

import random
import numpy as np
import pandas as pd
import multiprocessing
import time
import json
import os
from typing import Dict, List, Any, Callable, Union, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed

# Define parameter types
class ParameterType(Enum):
    """Enum for parameter types."""
    DISCRETE = "discrete"  # Discrete values (list of specific values)
    CONTINUOUS = "continuous"  # Continuous range (min, max, step)
    CATEGORICAL = "categorical"  # Categorical values (list of categories)
    BOOLEAN = "boolean"  # Boolean values (True/False)

@dataclass
class Parameter:
    """Parameter definition for genetic algorithm."""
    name: str
    param_type: ParameterType
    values: Union[List[Any], Tuple[float, float, float]]  # Values or (min, max, step) for continuous
    
    def sample(self) -> Any:
        """Sample a random value from the parameter space."""
        if self.param_type == ParameterType.DISCRETE:
            return random.choice(self.values)
        elif self.param_type == ParameterType.CONTINUOUS:
            min_val, max_val, step = self.values
            if step == 0:  # Truly continuous
                return random.uniform(min_val, max_val)
            else:  # Discretized continuous
                steps = int((max_val - min_val) / step) + 1
                return min_val + random.randint(0, steps - 1) * step
        elif self.param_type == ParameterType.CATEGORICAL:
            return random.choice(self.values)
        elif self.param_type == ParameterType.BOOLEAN:
            return random.choice([True, False])
        else:
            raise ValueError(f"Unknown parameter type: {self.param_type}")

@dataclass
class Individual:
    """Individual in the genetic algorithm population."""
    params: Dict[str, Any]
    fitness: Optional[float] = None
    metrics: Optional[Dict[str, float]] = None
    
    def __lt__(self, other):
        """Less than operator for sorting individuals by fitness."""
        if self.fitness is None:
            return True
        if other.fitness is None:
            return False
        return self.fitness < other.fitness

class SelectionMethod(Enum):
    """Enum for selection methods."""
    TOURNAMENT = "tournament"
    ROULETTE = "roulette"
    RANK = "rank"
    ELITISM = "elitism"

class CrossoverMethod(Enum):
    """Enum for crossover methods."""
    UNIFORM = "uniform"
    SINGLE_POINT = "single_point"
    TWO_POINT = "two_point"

class MutationMethod(Enum):
    """Enum for mutation methods."""
    UNIFORM = "uniform"
    GAUSSIAN = "gaussian"
    ADAPTIVE = "adaptive"

class GeneticAlgorithm:
    """
    Advanced Genetic Algorithm optimizer for trading strategy parameters.
    
    Features:
    - Support for different parameter types (continuous, discrete, categorical)
    - Multiple selection, crossover, and mutation methods
    - Parallel fitness evaluation
    - Detailed logging and visualization
    - Multi-objective optimization capabilities
    """
    
    def __init__(
        self,
        parameters: List[Parameter],
        fitness_func: Callable[[Dict[str, Any]], Union[float, Dict[str, float]]],
        population_size: int = 50,
        generations: int = 100,
        selection_method: SelectionMethod = SelectionMethod.TOURNAMENT,
        crossover_method: CrossoverMethod = CrossoverMethod.UNIFORM,
        mutation_method: MutationMethod = MutationMethod.UNIFORM,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.2,
        elitism_rate: float = 0.1,
        tournament_size: int = 3,
        multi_objective: bool = False,
        parallel: bool = True,
        max_workers: Optional[int] = None,
        log_dir: Optional[str] = None,
        random_seed: Optional[int] = None,
    ):
        """
        Initialize the genetic algorithm optimizer.
        
        Args:
            parameters: List of Parameter objects defining the parameter space
            fitness_func: Function that evaluates an individual and returns fitness score(s)
            population_size: Size of the population
            generations: Number of generations to evolve
            selection_method: Method for selecting parents
            crossover_method: Method for crossover
            mutation_method: Method for mutation
            crossover_rate: Probability of crossover
            mutation_rate: Probability of mutation
            elitism_rate: Percentage of top individuals to preserve
            tournament_size: Size of tournament for tournament selection
            multi_objective: Whether to use multi-objective optimization
            parallel: Whether to use parallel processing
            max_workers: Maximum number of parallel workers (None = auto)
            log_dir: Directory for logging results
            random_seed: Random seed for reproducibility
        """
        self.parameters = {param.name: param for param in parameters}
        self.fitness_func = fitness_func
        self.population_size = population_size
        self.generations = generations
        self.selection_method = selection_method
        self.crossover_method = crossover_method
        self.mutation_method = mutation_method
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elitism_rate = elitism_rate
        self.tournament_size = tournament_size
        self.multi_objective = multi_objective
        self.parallel = parallel
        self.max_workers = max_workers or max(1, multiprocessing.cpu_count() - 1)
        self.log_dir = log_dir
        
        # Set random seed if provided
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
        
        # Create log directory if specified
        if self.log_dir:
            os.makedirs(self.log_dir, exist_ok=True)
        
        # Initialize history
        self.history = {
            "best_fitness": [],
            "avg_fitness": [],
            "best_individual": [],
            "generation_time": []
        }
    
    def initialize_population(self) -> List[Individual]:
        """Initialize the population with random individuals."""
        population = []
        for _ in range(self.population_size):
            params = {name: param.sample() for name, param in self.parameters.items()}
            population.append(Individual(params=params))
        return population
    
    def evaluate_fitness(self, population: List[Individual]) -> List[Individual]:
        """Evaluate fitness for all individuals in the population."""
        if self.parallel and len(population) > 1:
            return self._evaluate_fitness_parallel(population)
        else:
            return self._evaluate_fitness_sequential(population)
    
    def _evaluate_fitness_sequential(self, population: List[Individual]) -> List[Individual]:
        """Evaluate fitness sequentially."""
        for individual in population:
            if individual.fitness is None:  # Only evaluate if not already evaluated
                result = self.fitness_func(individual.params)
                if isinstance(result, dict):
                    # Multi-objective fitness
                    individual.metrics = result
                    # Use primary objective as fitness if multi-objective
                    individual.fitness = result.get("primary_objective", 
                                                  result.get("sharpe_ratio", 
                                                           list(result.values())[0]))
                else:
                    # Single objective fitness
                    individual.fitness = result
                    individual.metrics = {"fitness": result}
        return population
    
    def _evaluate_fitness_parallel(self, population: List[Individual]) -> List[Individual]:
        """Evaluate fitness in parallel."""
        # Only evaluate individuals that haven't been evaluated yet
        to_evaluate = [ind for ind in population if ind.fitness is None]
        if not to_evaluate:
            return population
            
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all jobs
            future_to_individual = {
                executor.submit(self.fitness_func, ind.params): ind 
                for ind in to_evaluate
            }
            
            # Process results as they complete
            for future in as_completed(future_to_individual):
                individual = future_to_individual[future]
                try:
                    result = future.result()
                    if isinstance(result, dict):
                        # Multi-objective fitness
                        individual.metrics = result
                        # Use primary objective as fitness if multi-objective
                        individual.fitness = result.get("primary_objective", 
                                                      result.get("sharpe_ratio", 
                                                               list(result.values())[0]))
                    else:
                        # Single objective fitness
                        individual.fitness = result
                        individual.metrics = {"fitness": result}
                except Exception as e:
                    print(f"Error evaluating individual: {e}")
                    individual.fitness = float('-inf')
                    individual.metrics = {"error": str(e)}
        
        return population
    
    def select_parents(self, population: List[Individual]) -> Tuple[Individual, Individual]:
        """Select two parents from the population."""
        if self.selection_method == SelectionMethod.TOURNAMENT:
            return self._tournament_selection(population)
        elif self.selection_method == SelectionMethod.ROULETTE:
            return self._roulette_selection(population)
        elif self.selection_method == SelectionMethod.RANK:
            return self._rank_selection(population)
        else:
            raise ValueError(f"Unknown selection method: {self.selection_method}")
    
    def _tournament_selection(self, population: List[Individual]) -> Tuple[Individual, Individual]:
        """Tournament selection."""
        # First parent
        tournament = random.sample(population, self.tournament_size)
        parent1 = max(tournament, key=lambda ind: ind.fitness)
        
        # Second parent
        tournament = random.sample(population, self.tournament_size)
        parent2 = max(tournament, key=lambda ind: ind.fitness)
        
        return parent1, parent2
    
    def _roulette_selection(self, population: List[Individual]) -> Tuple[Individual, Individual]:
        """Roulette wheel selection."""
        # Adjust fitness values to be positive
        min_fitness = min(ind.fitness for ind in population)
        adjusted_fitness = [ind.fitness - min_fitness + 1e-10 for ind in population]
        total_fitness = sum(adjusted_fitness)
        
        # Calculate selection probabilities
        probabilities = [fit / total_fitness for fit in adjusted_fitness]
        
        # Select two parents
        indices = np.random.choice(len(population), size=2, p=probabilities)
        return population[indices[0]], population[indices[1]]
    
    def _rank_selection(self, population: List[Individual]) -> Tuple[Individual, Individual]:
        """Rank-based selection."""
        # Sort population by fitness
        sorted_pop = sorted(population, key=lambda ind: ind.fitness, reverse=True)
        
        # Assign ranks (higher rank = higher fitness)
        ranks = list(range(1, len(sorted_pop) + 1))
        total_rank = sum(ranks)
        
        # Calculate selection probabilities
        probabilities = [rank / total_rank for rank in ranks]
        
        # Select two parents
        indices = np.random.choice(len(sorted_pop), size=2, p=probabilities)
        return sorted_pop[indices[0]], sorted_pop[indices[1]]
    
    def crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """Perform crossover between two parents."""
        if random.random() > self.crossover_rate:
            # No crossover, return copies of parents
            return (
                Individual(params=parent1.params.copy()),
                Individual(params=parent2.params.copy())
            )
        
        if self.crossover_method == CrossoverMethod.UNIFORM:
            return self._uniform_crossover(parent1, parent2)
        elif self.crossover_method == CrossoverMethod.SINGLE_POINT:
            return self._single_point_crossover(parent1, parent2)
        elif self.crossover_method == CrossoverMethod.TWO_POINT:
            return self._two_point_crossover(parent1, parent2)
        else:
            raise ValueError(f"Unknown crossover method: {self.crossover_method}")
    
    def _uniform_crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """Uniform crossover."""
        child1_params = {}
        child2_params = {}
        
        for param_name in self.parameters:
            if random.random() < 0.5:
                child1_params[param_name] = parent1.params[param_name]
                child2_params[param_name] = parent2.params[param_name]
            else:
                child1_params[param_name] = parent2.params[param_name]
                child2_params[param_name] = parent1.params[param_name]
        
        return Individual(params=child1_params), Individual(params=child2_params)
    
    def _single_point_crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """Single-point crossover."""
        param_names = list(self.parameters.keys())
        crossover_point = random.randint(1, len(param_names) - 1)
        
        child1_params = {}
        child2_params = {}
        
        for i, param_name in enumerate(param_names):
            if i < crossover_point:
                child1_params[param_name] = parent1.params[param_name]
                child2_params[param_name] = parent2.params[param_name]
            else:
                child1_params[param_name] = parent2.params[param_name]
                child2_params[param_name] = parent1.params[param_name]
        
        return Individual(params=child1_params), Individual(params=child2_params)
    
    def _two_point_crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """Two-point crossover."""
        param_names = list(self.parameters.keys())
        if len(param_names) < 3:
            return self._single_point_crossover(parent1, parent2)
            
        point1 = random.randint(1, len(param_names) - 2)
        point2 = random.randint(point1 + 1, len(param_names) - 1)
        
        child1_params = {}
        child2_params = {}
        
        for i, param_name in enumerate(param_names):
            if i < point1 or i >= point2:
                child1_params[param_name] = parent1.params[param_name]
                child2_params[param_name] = parent2.params[param_name]
            else:
                child1_params[param_name] = parent2.params[param_name]
                child2_params[param_name] = parent1.params[param_name]
        
        return Individual(params=child1_params), Individual(params=child2_params)
    
    def mutate(self, individual: Individual) -> Individual:
        """Mutate an individual."""
        if random.random() > self.mutation_rate:
            return individual  # No mutation
        
        if self.mutation_method == MutationMethod.UNIFORM:
            return self._uniform_mutation(individual)
        elif self.mutation_method == MutationMethod.GAUSSIAN:
            return self._gaussian_mutation(individual)
        elif self.mutation_method == MutationMethod.ADAPTIVE:
            return self._adaptive_mutation(individual)
        else:
            raise ValueError(f"Unknown mutation method: {self.mutation_method}")
    
    def _uniform_mutation(self, individual: Individual) -> Individual:
        """Uniform mutation - randomly replace parameters with new values."""
        mutated_params = individual.params.copy()
        
        # Select a random parameter to mutate
        param_name = random.choice(list(self.parameters.keys()))
        param = self.parameters[param_name]
        
        # Replace with a new random value
        mutated_params[param_name] = param.sample()
        
        return Individual(params=mutated_params)
    
    def _gaussian_mutation(self, individual: Individual) -> Individual:
        """Gaussian mutation - add Gaussian noise to continuous parameters."""
        mutated_params = individual.params.copy()
        
        # Select a random parameter to mutate
        param_name = random.choice(list(self.parameters.keys()))
        param = self.parameters[param_name]
        
        if param.param_type == ParameterType.CONTINUOUS:
            # Add Gaussian noise for continuous parameters
            min_val, max_val, step = param.values
            value = mutated_params[param_name]
            
            # Scale the noise based on the parameter range
            sigma = (max_val - min_val) * 0.1
            noise = random.gauss(0, sigma)
            
            # Apply noise and clip to valid range
            new_value = value + noise
            new_value = max(min_val, min(max_val, new_value))
            
            # Discretize if step > 0
            if step > 0:
                new_value = min_val + round((new_value - min_val) / step) * step
                
            mutated_params[param_name] = new_value
        else:
            # For non-continuous parameters, use uniform mutation
            mutated_params[param_name] = param.sample()
        
        return Individual(params=mutated_params)
    
    def _adaptive_mutation(self, individual: Individual) -> Individual:
        """Adaptive mutation - mutation rate adapts based on generation."""
        # This is a placeholder for adaptive mutation
        # In a real implementation, this would adjust mutation based on progress
        return self._uniform_mutation(individual)
    
    def evolve(self) -> Tuple[Individual, pd.DataFrame]:
        """
        Evolve the population for the specified number of generations.
        
        Returns:
            Tuple containing the best individual and a DataFrame with optimization history
        """
        # Initialize population
        population = self.initialize_population()
        
        # Evaluate initial population
        population = self.evaluate_fitness(population)
        
        # Track best individual
        best_individual = max(population, key=lambda ind: ind.fitness)
        
        # Log initial state
        self._log_generation(0, population, best_individual, 0)
        
        # Evolution loop
        for generation in range(1, self.generations + 1):
            start_time = time.time()
            
            # Sort population by fitness
            population.sort(key=lambda ind: ind.fitness, reverse=True)
            
            # Elitism - keep best individuals
            elite_count = max(1, int(self.population_size * self.elitism_rate))
            elite = population[:elite_count]
            
            # Create new population
            new_population = elite.copy()
            
            # Fill the rest of the population with offspring
            while len(new_population) < self.population_size:
                # Select parents
                parent1, parent2 = self.select_parents(population)
                
                # Crossover
                child1, child2 = self.crossover(parent1, parent2)
                
                # Mutation
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                
                # Add to new population
                new_population.append(child1)
                if len(new_population) < self.population_size:
                    new_population.append(child2)
            
            # Evaluate new population
            population = self.evaluate_fitness(new_population)
            
            # Update best individual
            current_best = max(population, key=lambda ind: ind.fitness)
            if current_best.fitness > best_individual.fitness:
                best_individual = current_best
            
            # Calculate generation time
            generation_time = time.time() - start_time
            
            # Log generation
            self._log_generation(generation, population, best_individual, generation_time)
            
            # Print progress
            if generation % 10 == 0 or generation == 1 or generation == self.generations:
                self._print_progress(generation, population, best_individual, generation_time)
        
        # Create history DataFrame
        history_df = pd.DataFrame({
            "generation": list(range(self.generations + 1)),
            "best_fitness": self.history["best_fitness"],
            "avg_fitness": self.history["avg_fitness"],
            "generation_time": self.history["generation_time"]
        })
        
        # Save final results
        self._save_results(best_individual, history_df)
        
        return best_individual, history_df
    
    def _log_generation(self, generation: int, population: List[Individual], 
                       best_individual: Individual, generation_time: float):
        """Log generation statistics."""
        # Calculate average fitness
        valid_fitness = [ind.fitness for ind in population if ind.fitness is not None]
        avg_fitness = sum(valid_fitness) / len(valid_fitness) if valid_fitness else 0
        
        # Update history
        self.history["best_fitness"].append(best_individual.fitness)
        self.history["avg_fitness"].append(avg_fitness)
        self.history["best_individual"].append(best_individual.params.copy())
        self.history["generation_time"].append(generation_time)
    
    def _print_progress(self, generation: int, population: List[Individual], 
                       best_individual: Individual, generation_time: float):
        """Print progress information."""
        avg_fitness = self.history["avg_fitness"][-1]
        
        print(f"Generation {generation}/{self.generations} - "
              f"Best Fitness: {best_individual.fitness:.4f}, "
              f"Avg Fitness: {avg_fitness:.4f}, "
              f"Time: {generation_time:.2f}s")
        
        # Print best individual parameters
        print(f"Best Parameters: {best_individual.params}")
        
        # Print metrics if available
        if best_individual.metrics:
            print("Metrics:")
            for key, value in best_individual.metrics.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")
    
    def _save_results(self, best_individual: Individual, history_df: pd.DataFrame):
        """Save optimization results."""
        if not self.log_dir:
            return
            
        # Create timestamp
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        
        # Save best individual
        best_params_file = os.path.join(self.log_dir, f"best_params_{timestamp}.json")
        with open(best_params_file, "w") as f:
            json.dump({
                "params": best_individual.params,
                "fitness": best_individual.fitness,
                "metrics": best_individual.metrics
            }, f, indent=2)
        
        # Save history
        history_file = os.path.join(self.log_dir, f"optimization_history_{timestamp}.csv")
        history_df.to_csv(history_file, index=False)
        
        # Create plots
        self._create_plots(history_df, timestamp)
    
    def _create_plots(self, history_df: pd.DataFrame, timestamp: str):
        """Create and save plots of optimization progress."""
        if not self.log_dir:
            return
            
        # Create fitness plot
        plt.figure(figsize=(10, 6))
        plt.plot(history_df["generation"], history_df["best_fitness"], label="Best Fitness")
        plt.plot(history_df["generation"], history_df["avg_fitness"], label="Average Fitness")
        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        plt.title("Optimization Progress")
        plt.legend()
        plt.grid(True)
        
        # Save plot
        plot_file = os.path.join(self.log_dir, f"optimization_progress_{timestamp}.png")
        plt.savefig(plot_file)
        plt.close()

def create_parameter_space(param_dict: Dict[str, Any]) -> List[Parameter]:
    """
    Create a list of Parameter objects from a dictionary specification.
    
    Args:
        param_dict: Dictionary with parameter specifications
            Format: {
                "param_name": {"type": "discrete", "values": [1, 2, 3]},
                "param_name2": {"type": "continuous", "min": 0, "max": 1, "step": 0.1}
            }
    
    Returns:
        List of Parameter objects
    """
    parameters = []
    
    for param_name, spec in param_dict.items():
        param_type_str = spec["type"].lower()
        
        if param_type_str == "discrete":
            param_type = ParameterType.DISCRETE
            values = spec["values"]
        elif param_type_str == "continuous":
            param_type = ParameterType.CONTINUOUS
            values = (spec["min"], spec["max"], spec.get("step", 0))
        elif param_type_str == "categorical":
            param_type = ParameterType.CATEGORICAL
            values = spec["values"]
        elif param_type_str == "boolean":
            param_type = ParameterType.BOOLEAN
            values = [True, False]
        else:
            raise ValueError(f"Unknown parameter type: {param_type_str}")
        
        parameters.append(Parameter(
            name=param_name,
            param_type=param_type,
            values=values
        ))
    
    return parameters
