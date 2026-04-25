import importlib
import yaml
import time

from src.pattern import Pattern
from src.ea import EvolutionaryAlgorithm
from src.utils import PROJECT_ROOT, INPUTS_DIR, OUTPUTS_DIR, consolidate_results, StatusPrinter
from src.visualization import save_ca_gif

def main():
    # Load configuration
    with open(PROJECT_ROOT / "configs" / "config.yaml") as f:
        config = yaml.safe_load(f)
    
    # Configuration extraction
    VERSION = config["output"]
    NUM_EXPR = config["num_experiments"]
    STEPS = config["steps"]
    GENERATIONS = config["generations"]
    patterns = config["patterns"]
    
    # Paths Setup
    VERSION_DIR = OUTPUTS_DIR / VERSION
    CSV_DIR = VERSION_DIR / "csv"
    SOLN_DIR = VERSION_DIR / "solns"
    GIF_DIR = VERSION_DIR / "gifs"

    # Load fitness
    fitness_module = importlib.import_module(f"src.fitness.{config['fitness']}")
    fitness_fn = fitness_module.compute_fitness

    # Status tracking
    status = StatusPrinter(
        total_experiments=len(patterns) * NUM_EXPR,
        total_generations=GENERATIONS,
        start_time=time.time()
    )
    global_expr = 0

    # Execution Loop
    for p_name in patterns:
        p_obj = Pattern.from_file(INPUTS_DIR / f"{p_name}.txt")

        for expr in range(NUM_EXPR):
            def make_callback(g_idx):
                return lambda **kwargs: status.update(
                    experiment=g_idx, generation=kwargs["generation"],
                    best_fit=kwargs["best_fit"], best_expr=g_idx,
                    best_gen=kwargs["best_gen"], pattern=kwargs["pattern"]
                )

            ea = EvolutionaryAlgorithm(
                pattern=p_obj, fitness_fn=fitness_fn, steps=STEPS,
                generations=GENERATIONS, N=config["N"],
                mutation_rate=config["mutation_rate"],
                crossover_rate=config["crossover_rate"],
                tournament_size=config["tournament_size"],
                seed=config["seed"]
            )

            # Specific file paths
            res_path = CSV_DIR / p_name / f"{p_name}_expr{expr}.csv"
            sln_path = SOLN_DIR / p_name / f"{p_name}_expr{expr}.json"

            ea.run(res_path, sln_path, expr, callback=make_callback(global_expr))
            global_expr += 1

    # Consolidate CSVs
    print("\nProcessing finished. Cleaning up and generating visualizations...")    
    consolidate_results(CSV_DIR, f"{VERSION}.csv")

if __name__ == "__main__":
    main()