import ast
import re
import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.utils import PROJECT_ROOT

INPUT_FILES = {
    "pixel_iou": PROJECT_ROOT / "outputs" / "pixel_iou" / "pixel_iou.csv",
    "boundary_shape": PROJECT_ROOT / "outputs" / "boundary_shape" / "boundary_shape.csv",
    "distance_morphology": PROJECT_ROOT / "outputs" / "distance_morphology" / "distance_morphology.csv",
}
SUMMARY_CSV = Path(__file__).parent / "best_individuals_summary.csv"

def parse_gen_column(s: str) -> list:
    cleaned = re.sub(r'np\.float64\(([^)]*)\)', r'\1', str(s))
    return ast.literal_eval(cleaned)

def find_best_individual_info(row):
    """Extracts best_fitness and best_step from the generation's lists."""
    fitnesses = parse_gen_column(row["gen_fitnesses"])
    steps = parse_gen_column(row["gen_steps"])
    best_idx = int(np.argmax(fitnesses))
    return float(fitnesses[best_idx]), int(steps[best_idx])

def main():
    all_summaries = []

    for metric, path in INPUT_FILES.items():
        if not path.exists():
            print(f"Skipping {metric}: File not found.")
            continue

        print(f"Analyzing {metric}...")
        df = pd.read_csv(path)
        
        # Find first row where best_overall is max per pattern
        for name, group in df.groupby("name", sort=False):
            max_val = group["best_overall"].max()
            best_row = group[group["best_overall"] == max_val].iloc[0].copy()
            
            # Get the specific individual's stats from that generation
            best_fit, best_step = find_best_individual_info(best_row)
            
            all_summaries.append({
                "fitness_method": metric,
                "pattern_name": name,
                "experiment": int(best_row["experiment"]),
                "generation": int(best_row["generation"]),
                "best_fitness": best_fit,
                "best_step": best_step
            })

    summary_df = pd.DataFrame(all_summaries)
    summary_df.to_csv(SUMMARY_CSV, index=False)
    print(f"\nDone! Summary saved to: {SUMMARY_CSV}")

if __name__ == "__main__":
    main()