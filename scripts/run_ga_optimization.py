"""
CLI tool to run GA optimization experiments.
"""

import json
from src.strategies.ga_optimizer import GAOptimizer
from examples.ga_optimize_ma_crossover import run_backtest_with_params

param_space = {
    "fast_period": list(range(5, 30)),
    "slow_period": list(range(20, 100)),
    "threshold": [0.0, 0.01, 0.02, 0.05]
}

def main():
    runs = 3
    best_results = []

    for i in range(runs):
        print(f"Starting GA run {i+1}/{runs}")
        optimizer = GAOptimizer(
            param_space=param_space,
            fitness_func=run_backtest_with_params,
            population_size=10,
            generations=20,
            crossover_rate=0.7,
            mutation_rate=0.2
        )
        best_params = optimizer.evolve()
        best_results.append(best_params)
        print(f"Run {i+1} best params: {best_params}")

    # Save results
    with open("ga_optimization_results.json", "w") as f:
        json.dump(best_results, f, indent=2)

    print("GA optimization completed. Results saved to ga_optimization_results.json")

if __name__ == "__main__":
    main()