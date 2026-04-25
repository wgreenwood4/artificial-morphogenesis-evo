import numpy as np
import random
import copy

from .ca import CellularAutomata
from .pattern import Pattern
from .utils import create_seed, EvolutionLogger

class EvolutionaryAlgorithm:
    def __init__(
            self,
            pattern: Pattern,
            fitness_fn,
            steps=50,
            generations=100,
            N=50,
            mutation_rate=0.02,
            crossover_rate=0.8,
            tournament_size=3,
            seed=420,
            initial_population=None
        ):
        
        self.pattern = pattern
        self.fitness_fn = fitness_fn
        self.steps = steps
        self.generations = generations
        self.N = N
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
        self.seed = seed
        self.initial_population = initial_population

    def initialize_population(self):
        population = []
        
        if self.initial_population is not None:
            population = [
                copy.deepcopy(ca) for ca in self.initial_population[:self.N]
            ]

        remaining = self.N - len(population)

        for _ in range(remaining):
            population.append(CellularAutomata())

        return population

    def select(self, population, fitnesses):
        sel_indices = [
            random.randint(0, len(population)-1) 
            for _ in range(self.tournament_size)
        ]
        sel_fits = [fitnesses[i] for i in sel_indices]
        best_index = sel_indices[sel_fits.index(max(sel_fits))]
        return population[best_index]

    def crossover(self, p1, p2):
        child_rules = np.where(
            np.random.rand(len(p1.rules)) < 0.5,
            p1.rules,
            p2.rules
        ).astype(np.uint8)
        return CellularAutomata(child_rules)

    def mutate(self, ca):
        for i in range(1, len(ca.rules)):
            if random.random() < self.mutation_rate:
                ca.rules[i] = 1 - ca.rules[i]
        return ca

    def run(self, results_path, solution_path, expr, callback=None):
        best_ca = None
        best_fit = -1.0
        best_gen = 0

        with EvolutionLogger(results_path, solution_path, seed=self.seed) as logger:
            current_seed = self.seed + expr
            random.seed(current_seed)
            np.random.seed(current_seed)

            population = self.initialize_population()
            seed_grid = create_seed(self.pattern.grid.shape)

            for generation in range(self.generations):
                best_fitnesses = []
                best_steps = []
                
                for ca in population:
                    grid = seed_grid.copy()
                    best_fitness = 0.0
                    best_step = 0

                    # Step through CA and take highest fitness and store that step
                    eval_interval = 2
                    for step in range(self.steps):
                        prev = grid
                        grid = ca.step(grid)

                        if step % eval_interval == 0:
                            fitness = self.fitness_fn(grid, self.pattern.grid, prev)
                            
                            if fitness > best_fitness:
                                best_fitness = fitness
                                best_step = step + 1
                            
                            if best_fitness > 0.999:
                                break

                    best_fitnesses.append(best_fitness)
                    best_steps.append(best_step)
                
                idx_best = int(np.argmax(best_fitnesses))
                idx_worst = int(np.argmin(best_fitnesses))

                # Best/worst fitnesses in generation
                best_fit_in_gen = float(best_fitnesses[idx_best])
                worst_fit_in_gen = float(best_fitnesses[idx_worst])
                best_ca_in_gen = copy.deepcopy(population[idx_best])

                # Global tracker
                if best_fit_in_gen > best_fit:
                    best_fit = best_fit_in_gen
                    best_ca = copy.deepcopy(best_ca_in_gen)
                    best_gen = generation
                
                avg_in_gen = float(np.mean(best_fitnesses))
                
                logger.log({
                    "name": self.pattern.name,
                    "experiment": expr,
                    "generation": generation,
                    "best_in_gen": best_fit_in_gen,
                    "worst_in_gen": worst_fit_in_gen,
                    "avg_in_gen": avg_in_gen,
                    "best_overall": best_fit,
                    "gen_fitnesses": best_fitnesses,
                    "gen_steps": best_steps,
                    "seed": current_seed
                })

                if callback:
                    callback(
                        experiment=expr,
                        generation=generation,
                        best_fit=best_fit,
                        best_gen=best_gen,
                        pattern=self.pattern.name
                    )
                
                # Build new population
                elite_indices = np.argsort(best_fitnesses)[-3:]
                new_population = [
                    copy.deepcopy(population[i])
                    for i in elite_indices
                ]
                while len(new_population) < self.N:
                    p1 = self.select(population, best_fitnesses)
                    p2 = self.select(population, best_fitnesses)
                
                    # Crossover
                    if random.random() < self.crossover_rate:
                        child = self.crossover(p1, p2)
                    else:
                        child = CellularAutomata(p1.rules.copy())

                    # Mutation
                    child = self.mutate(child)
                    new_population.append(child)
                
                population = new_population

            # Log final best solution
            logger.log_solution(
                best_ca,
                best_fit,
                metadata={
                    "pattern": self.pattern.name,
                    "shape": self.pattern.shape,
                    "steps": self.steps,
                    "generations": self.generations,
                    "population": self.N,
                    "mutation_rate": self.mutation_rate,
                    "crossover_rate": self.crossover_rate,
                    "tournament_size": self.tournament_size,
                    "experiment": expr,
                }
            )
        
        return best_fit
