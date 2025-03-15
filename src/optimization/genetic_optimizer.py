"""
Genetic algorithm-based optimizer for trading strategy parameters.

This module provides a framework for optimizing trading strategy parameters using 
genetic algorithms, which are well-suited for finding optimal solutions in complex, 
multi-dimensional parameter spaces.
"""
import numpy as np
import pandas as pd
import random
from typing import Dict, List, Callable, Any, Tuple, Optional
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed
import time


class GeneticOptimizer:
    """
    Genetic algorithm-based optimizer for trading strategy parameters.
    
    This optimizer uses genetic algorithms to find optimal parameter sets for trading strategies,
    allowing for efficient exploration of complex parameter spaces to maximize objectives like
    returns, Sharpe ratio, or custom fitness functions.
    """
    
    def __init__(
        self,
        param_bounds: Dict[str, Tuple[float, float]],
        fitness_function: Callable,
        population_size: int = 50,
        generations: int = 20,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.2,
        elite_ratio: float = 0.1,
        discretize_params: Optional[Dict[str, int]] = None,
        maximize: bool = True,
        parallel: bool = True,
        max_workers: int = None
    ):
        """
        Initialize the genetic optimizer.
        
        Args:
            param_bounds: Dictionary mapping parameter names to (min, max) bounds
            fitness_function: Function that evaluates parameter sets and returns a fitness score
            population_size: Size of the population in each generation
            generations: Number of generations to evolve
            crossover_rate: Probability of crossover between individuals
            mutation_rate: Probability of mutation for each gene
            elite_ratio: Percentage of top performers to preserve for next generation
            discretize_params: Dictionary mapping parameter names to number of discrete steps
            maximize: If True, try to maximize the fitness function; if False, minimize
            parallel: If True, use parallel processing for fitness evaluation
            max_workers: Maximum number of worker processes for parallel execution
        """
        self.param_bounds = param_bounds
        self.fitness_function = fitness_function
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elite_ratio = elite_ratio
        self.discretize_params = discretize_params or {}
        self.maximize = maximize
        self.parallel = parallel
        self.max_workers = max_workers
        
        # Store optimization results
        self.best_individual = None
        self.best_fitness = float('-inf') if maximize else float('inf')
        self.history = {
            'best_fitness': [],
            'avg_fitness': [],
            'best_params': []
        }
    
    def _generate_individual(self):
        """
        Generate a random individual with parameters within the specified bounds.
        
        Returns:
            Dict with randomized parameter values
        """
        individual = {}
        for param, bounds in self.param_bounds.items():
            # Handle the case where bounds is a list of choices (strings, booleans)
            if isinstance(bounds, (list, tuple)) and len(bounds) > 0:
                if all(isinstance(x, str) for x in bounds):
                    # List of string choices
                    individual[param] = random.choice(bounds)
                    continue
                elif all(isinstance(x, bool) for x in bounds):
                    # List of boolean choices
                    individual[param] = random.choice(bounds)
                    continue
                elif len(bounds) == 2:
                    # Possibly a min-max range
                    min_val, max_val = bounds
                else:
                    # Other list type
                    individual[param] = random.choice(bounds)
                    continue
            elif not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
                raise ValueError(f"Parameter bounds for '{param}' must be a tuple/list of (min, max) values or a list of choices.")
            else:
                # Regular min-max bounds
                min_val, max_val = bounds
            
            # Generate random value based on the parameter type
            if isinstance(min_val, int) and isinstance(max_val, int):
                # Integer parameter
                individual[param] = random.randint(min_val, max_val)
            elif isinstance(min_val, bool) or isinstance(max_val, bool):
                # Boolean parameter
                individual[param] = random.choice([True, False])
            else:
                # Float parameter
                individual[param] = min_val + random.random() * (max_val - min_val)
                
        return individual
    
    def _initialize_population(self) -> List[Dict[str, float]]:
        """Initialize a random population of individuals."""
        return [self._generate_individual() for _ in range(self.population_size)]
    
    def _evaluate_population(self, population: List[Dict[str, float]]) -> List[float]:
        """Evaluate fitness for each individual in the population."""
        if self.parallel:
            return self._evaluate_population_parallel(population)
        else:
            return [self.fitness_function(individual) for individual in population]
    
    def _evaluate_population_parallel(self, population: List[Dict[str, float]]) -> List[float]:
        """Evaluate fitness for each individual using parallel processing."""
        fitness_values = []
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self.fitness_function, individual): i 
                      for i, individual in enumerate(population)}
            
            for future in as_completed(futures):
                fitness_values.append((futures[future], future.result()))
        
        # Sort by original index and extract just the fitness values
        fitness_values.sort(key=lambda x: x[0])
        return [f for _, f in fitness_values]
    
    def _select_parent(self, population: List[Dict[str, float]], fitness_values: List[float]) -> Dict[str, float]:
        """Select a parent using tournament selection."""
        # Tournament selection
        tournament_size = 3
        tournament_indices = random.sample(range(len(population)), tournament_size)
        tournament_fitness = [fitness_values[i] for i in tournament_indices]
        
        if self.maximize:
            winner_idx = tournament_indices[tournament_fitness.index(max(tournament_fitness))]
        else:
            winner_idx = tournament_indices[tournament_fitness.index(min(tournament_fitness))]
            
        return population[winner_idx]
    
    def _crossover(self, parent1, parent2):
        """Perform crossover between two parents to create a child."""
        child = {}
        
        for param in parent1:
            # Different crossover approaches based on parameter type
            if isinstance(parent1[param], bool):
                # For booleans, randomly select from either parent
                child[param] = parent1[param] if random.random() < 0.5 else parent2[param]
            elif isinstance(parent1[param], str):
                # For strings, take from either parent
                child[param] = parent1[param] if random.random() < 0.5 else parent2[param]
            else:
                # For numerical values, use crossover probability
                if random.random() < self.crossover_rate:
                    # Perform interpolation between values
                    alpha = random.random()  # Random weight
                    child[param] = parent1[param] * alpha + parent2[param] * (1 - alpha)
                else:
                    # Just take value from one parent
                    child[param] = parent1[param] if random.random() < 0.5 else parent2[param]
                    
                    # Apply discretization if specified
                    if param in self.discretize_params:
                        bounds = self.param_bounds[param]
                        if isinstance(bounds, (list, tuple)) and len(bounds) == 2:
                            min_val, max_val = bounds
                            num_steps = self.discretize_params[param]
                            step_size = (max_val - min_val) / num_steps
                            child[param] = min_val + round((child[param] - min_val) / step_size) * step_size
        
        return child
        
    def _mutate(self, individual):
        """Apply random mutations to an individual."""
        mutated = individual.copy()
        
        for param in mutated:
            # Only mutate with the specified probability
            if random.random() < self.mutation_rate:
                bounds = self.param_bounds[param]
                
                # Handle mutation differently based on parameter type
                if isinstance(individual[param], bool):
                    # Flip boolean value
                    mutated[param] = not individual[param]
                elif isinstance(individual[param], str):
                    # For string choices, select a different option
                    if isinstance(bounds, (list, tuple)) and all(isinstance(x, str) for x in bounds):
                        choices = [x for x in bounds if x != individual[param]]
                        if choices:  # Ensure there are other choices available
                            mutated[param] = random.choice(choices)
                elif isinstance(bounds, (list, tuple)) and len(bounds) == 2:
                    # Handle numerical parameters
                    min_val, max_val = bounds
                    
                    # Different mutation strategy for integers and floats
                    if isinstance(min_val, int) and isinstance(max_val, int):
                        # For integers, generate a new random int
                        mutated[param] = random.randint(min_val, max_val)
                    else:
                        # For floats, perturb the current value
                        mutation_strength = (max_val - min_val) * 0.1  # 10% of range
                        delta = random.uniform(-mutation_strength, mutation_strength)
                        mutated[param] = max(min_val, min(max_val, individual[param] + delta))
                        
                        # Apply discretization if specified
                        if param in self.discretize_params:
                            num_steps = self.discretize_params[param]
                            step_size = (max_val - min_val) / num_steps
                            mutated[param] = min_val + round((mutated[param] - min_val) / step_size) * step_size
                
        return mutated
    
    def optimize(self, verbose: bool = True) -> Dict[str, Any]:
        """
        Run the genetic optimization algorithm.
        
        Args:
            verbose: If True, print progress information
            
        Returns:
            Dictionary with optimization results
        """
        start_time = time.time()
        
        # Initialize population
        population = self._initialize_population()
        
        for generation in range(self.generations):
            gen_start_time = time.time()
            
            # Evaluate fitness
            fitness_values = self._evaluate_population(population)
            
            # Track best individual
            best_idx = fitness_values.index(max(fitness_values) if self.maximize else min(fitness_values))
            generation_best = population[best_idx]
            generation_best_fitness = fitness_values[best_idx]
            
            # Update all-time best
            if ((self.maximize and generation_best_fitness > self.best_fitness) or 
                (not self.maximize and generation_best_fitness < self.best_fitness)):
                self.best_fitness = generation_best_fitness
                self.best_individual = generation_best.copy()
            
            # Store history
            self.history['best_fitness'].append(generation_best_fitness)
            self.history['avg_fitness'].append(sum(fitness_values) / len(fitness_values))
            self.history['best_params'].append(generation_best.copy())
            
            if verbose and (generation % 5 == 0 or generation == self.generations - 1):
                elapsed = time.time() - gen_start_time
                print(f"Generation {generation+1}/{self.generations} | "
                      f"Best fitness: {generation_best_fitness:.4f} | "
                      f"Avg fitness: {self.history['avg_fitness'][-1]:.4f} | "
                      f"Time: {elapsed:.2f}s")
            
            # Exit if last generation
            if generation == self.generations - 1:
                break
            
            # Create next generation
            next_population = []
            
            # Elitism - keep the best performers
            elite_count = int(self.population_size * self.elite_ratio)
            if elite_count > 0:
                # Sort indices by fitness values
                sorted_indices = sorted(range(len(fitness_values)), 
                                       key=lambda i: fitness_values[i],
                                       reverse=self.maximize)
                
                # Add the top performers to the next generation
                for i in range(elite_count):
                    next_population.append(population[sorted_indices[i]].copy())
            
            # Fill the rest with offspring from crossover and mutation
            while len(next_population) < self.population_size:
                parent1 = self._select_parent(population, fitness_values)
                parent2 = self._select_parent(population, fitness_values)
                
                child = self._crossover(parent1, parent2)
                child = self._mutate(child)
                
                next_population.append(child)
            
            # Replace the old population
            population = next_population
        
        total_time = time.time() - start_time
        
        if verbose:
            print(f"Optimization completed in {total_time:.2f}s")
            print(f"Best parameters: {self.best_individual}")
            print(f"Best fitness: {self.best_fitness:.4f}")
        
        return {
            'best_params': self.best_individual,
            'best_fitness': self.best_fitness,
            'history': self.history,
            'execution_time': total_time
        }
    
    def plot_progress(self, figsize=(10, 6)):
        """Plot the optimization progress over generations."""
        if not self.history['best_fitness']:
            raise ValueError("No optimization history available. Run optimize() first.")
        
        plt.figure(figsize=figsize)
        generations = list(range(1, len(self.history['best_fitness']) + 1))
        
        plt.plot(generations, self.history['best_fitness'], 'b-', label='Best Fitness')
        plt.plot(generations, self.history['avg_fitness'], 'r--', label='Average Fitness')
        
        plt.title('Genetic Algorithm Optimization Progress')
        plt.xlabel('Generation')
        plt.ylabel('Fitness Score')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
    def get_top_n_solutions(self, n: int = 10) -> pd.DataFrame:
        """
        Get the top N parameter sets and their fitness values.
        
        Args:
            n: Number of top solutions to return
            
        Returns:
            DataFrame with top parameter sets and their fitness values
        """
        if not self.history['best_params']:
            raise ValueError("No optimization history available. Run optimize() first.")
        
        # Create a list of all unique parameter sets and their fitness values
        all_params = []
        all_fitness = []
        
        # Use set to track unique parameter sets
        seen_params = set()
        
        for gen_idx, params in enumerate(self.history['best_params']):
            # Convert dict to tuple for hashing
            param_tuple = tuple(sorted((k, str(v)) for k, v in params.items()))
            
            if param_tuple not in seen_params:
                seen_params.add(param_tuple)
                all_params.append(params)
                all_fitness.append(self.history['best_fitness'][gen_idx])
        
        # Sort by fitness
        sorted_indices = sorted(range(len(all_fitness)), 
                               key=lambda i: all_fitness[i],
                               reverse=self.maximize)
        
        # Take top N
        top_n_indices = sorted_indices[:min(n, len(sorted_indices))]
        top_n_params = [all_params[i] for i in top_n_indices]
        top_n_fitness = [all_fitness[i] for i in top_n_indices]
        
        # Create DataFrame
        results = []
        for params, fitness in zip(top_n_params, top_n_fitness):
            result = params.copy()
            result['fitness'] = fitness
            results.append(result)
        
        return pd.DataFrame(results)
