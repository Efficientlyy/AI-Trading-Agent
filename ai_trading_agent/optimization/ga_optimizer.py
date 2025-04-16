import numpy as np
import random
from typing import Callable, List, Dict, Any

class GAOptimizer:
    """
    Basic Genetic Algorithm optimizer for trading strategy parameter optimization.
    """

    def __init__(
        self,
        fitness_fn: Callable[[Dict[str, Any]], float],
        param_space: Dict[str, List[Any]],
        population_size: int = 20,
        generations: int = 50,
        crossover_rate: float = 0.7,
        mutation_rate: float = 0.1,
        elite_fraction: float = 0.1,
        random_seed: int = 42,
    ):
        self.fitness_fn = fitness_fn
        self.param_space = param_space
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elite_fraction = elite_fraction
        self.random = random.Random(random_seed)

    def initialize_population(self) -> List[Dict[str, Any]]:
        population = []
        for _ in range(self.population_size):
            individual = {k: self.random.choice(v) for k, v in self.param_space.items()}
            population.append(individual)
        return population

    def evaluate_population(self, population: List[Dict[str, Any]]) -> List[float]:
        return [self.fitness_fn(ind) for ind in population]

    def select_parents(self, population: List[Dict[str, Any]], fitness: List[float]) -> List[Dict[str, Any]]:
        # Tournament selection
        selected = []
        for _ in range(len(population)):
            i, j = self.random.sample(range(len(population)), 2)
            winner = population[i] if fitness[i] > fitness[j] else population[j]
            selected.append(winner)
        return selected

    def crossover(self, parent1: Dict[str, Any], parent2: Dict[str, Any]) -> Dict[str, Any]:
        child = {}
        for k in self.param_space.keys():
            child[k] = parent1[k] if self.random.random() < 0.5 else parent2[k]
        return child

    def mutate(self, individual: Dict[str, Any]) -> Dict[str, Any]:
        mutant = individual.copy()
        for k, v in self.param_space.items():
            if self.random.random() < self.mutation_rate:
                mutant[k] = self.random.choice(v)
        return mutant

    def run(self) -> Dict[str, Any]:
        population = self.initialize_population()
        for gen in range(self.generations):
            fitness = self.evaluate_population(population)
            # Elitism
            elite_count = max(1, int(self.elite_fraction * self.population_size))
            elite_indices = np.argsort(fitness)[-elite_count:]
            elites = [population[i] for i in elite_indices]
            # Selection
            parents = self.select_parents(population, fitness)
            # Crossover and mutation
            next_population = elites.copy()
            while len(next_population) < self.population_size:
                if self.random.random() < self.crossover_rate:
                    p1, p2 = self.random.sample(parents, 2)
                    child = self.crossover(p1, p2)
                else:
                    child = self.random.choice(parents).copy()
                child = self.mutate(child)
                next_population.append(child)
            population = next_population[:self.population_size]
        # Final evaluation
        fitness = self.evaluate_population(population)
        best_idx = int(np.argmax(fitness))
        return population[best_idx]