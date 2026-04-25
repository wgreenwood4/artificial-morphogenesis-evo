# Helper functions and utilities

import yaml
import numpy as np
from pathlib import Path
import csv
import json
import time
import os

# Finds the root of the project relative to this file
PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUTS_DIR = PROJECT_ROOT / "inputs"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# Loads an input pattern into a numpy grid and yaml metadata 
def load_pattern(path: str):
    with open(path, "r") as f:
        content = f.read()
    
    parts = content.split('---')
    
    metadata = yaml.safe_load(parts[1])
    
    raw_grid = parts[2].strip()
    grid = np.loadtxt(raw_grid.splitlines(), dtype=int)
    
    return metadata, grid

def load_solution(path):
    with open(path, "r") as f:
        payload = json.load(f)

    rules = np.array(payload["rules"], dtype=np.uint8)
    fitness = payload["fitness"]
    metadata = payload.get("metadata", {})

    return rules, fitness, metadata

def create_seed(shape):
    grid = np.zeros(shape, dtype=int)
    grid[shape[0]//2, shape[1]//2] = 1
    return grid

class EvolutionLogger:
    fieldnames = [
        "name",
        "experiment",
        "generation",
        "best_in_gen",
        "worst_in_gen",
        "avg_in_gen",
        "best_overall",
        "gen_fitnesses",
        "gen_steps",
        "seed"
    ]

    def __init__(self, results_path, solution_path, seed=None):
        self.results_path = Path(results_path)
        self.solution_path = Path(solution_path)
        self.seed = seed
        
        self.results_path.parent.mkdir(parents=True, exist_ok=True)
        self.solution_path.parent.mkdir(parents=True, exist_ok=True)

        self.file = self.results_path.open("w", newline="")
        self.writer = csv.DictWriter(self.file, fieldnames=self.fieldnames)
        self.writer.writeheader()

    def log(self, row: dict):
        if "seed" not in row:
            row["seed"] = self.seed
        self.writer.writerow(row)
        self.file.flush()

    def log_solution(self, ca, fitness, metadata=None):
        payload = {
            "fitness": fitness,
            "rules": ca.rules.tolist(),
            "metadata": metadata
        }
        with self.solution_path.open("w") as f:
            json.dump(payload, f, indent=2)

    def close(self):
        self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

class StatusPrinter:
    def __init__(self, total_experiments, total_generations, start_time=None):
        self.total_experiments = total_experiments
        self.total_generations = total_generations
        self.start_time = start_time or time.time()
        self.gen_count = 0

        self.best_fit = -1
        self.best_expr = 0
        self.best_gen = 0

    def update(self, experiment, generation, best_fit, best_expr, best_gen, pattern):
        self.gen_count += 1

        # Track global best
        if best_fit > self.best_fit:
            self.best_fit = best_fit
            self.best_expr = best_expr
            self.best_gen = best_gen

        elapsed = time.time() - self.start_time
        total_gens = self.total_experiments * self.total_generations

        avg_time = elapsed / self.gen_count
        remaining = total_gens - self.gen_count
        eta = remaining * avg_time

        self.clear()

        print(f"PATTERN      {pattern}")
        print(f"EXPERIMENT   {experiment+1}/{self.total_experiments}")
        print(f"GENERATION   {generation+1}/{self.total_generations}")
        print()

        print(f"CURRENT BEST FIT   {best_fit:.4f}")
        print(f"GLOBAL BEST FIT    {self.best_fit:.4f}")
        print(f"FOUND AT           (expr {self.best_expr+1}, gen {self.best_gen+1})")
        print()

        print(f"TOTAL TIME ELAPSED {self.format_time(elapsed)}")
        print(f"AVG TIME / GEN     {avg_time:.3f}s")
        print(f"GLOBAL ETA         {self.format_time(eta)}")

    def clear(self):
        os.system("cls" if os.name == "nt" else "clear")

    def format_time(self, seconds):
        m, s = divmod(int(seconds), 60)
        h, m = divmod(m, 60)
        return f"{h:02}:{m:02}:{s:02}"

def consolidate_results(input_dir: str | Path, output_name: str):
    input_path = Path(input_dir)
    output_path = input_path / output_name
    
    # Grab all .csv files, excluding the output file if it already exists
    csv_files = [f for f in input_path.rglob("*.csv") if f.name != output_name]
    
    if not csv_files:
        print(f"No CSV files found in {input_path}")
        return

    print(f"Consolidating {len(csv_files)} files into {output_path}...")

    header_written = False
    
    with open(output_path, "w", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=EvolutionLogger.fieldnames)
        
        for csv_file in csv_files:
            with open(csv_file, "r") as f_in:
                reader = csv.DictReader(f_in)
                
                # Write the header only once
                if not header_written:
                    writer.writeheader()
                    header_written = True
                
                # Write all rows from the current file
                for row in reader:
                    writer.writerow(row)

    print("Consolidation complete.")
