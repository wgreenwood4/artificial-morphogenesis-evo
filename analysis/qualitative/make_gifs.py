import json
import sys
import pandas as pd
from pathlib import Path

# Add project root to path for imports
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.visualization import save_ca_gif
from src.utils import PROJECT_ROOT

# Paths setup
QUAL_DIR = Path(__file__).parent
SUMMARY_CSV = QUAL_DIR / "best_solns_summary.csv"
SOLUTIONS_ROOT = PROJECT_ROOT / "outputs"

def parse_args():
    if len(sys.argv) < 2:
        print("Usage: python make_gifs.py [best_step | max_step=40]")
        sys.exit(1)

    arg = sys.argv[1].lower()
    
    if arg == "best_step":
        return "best", None, QUAL_DIR / "best_step_gifs"
    
    if arg.startswith("max_step"):
        # Handle "max_step=40" or just "max_step" defaulting to 40
        try:
            val = int(arg.split("=")[1]) if "=" in arg else 40
        except (IndexError, ValueError):
            val = 40
        return "max", val, QUAL_DIR / "max_step_gifs"

    print("Invalid argument. Use 'best_step' or 'max_step=40'")
    sys.exit(1)

def main():
    mode, max_val, output_root = parse_args()

    if not SUMMARY_CSV.exists():
        print(f"Error: {SUMMARY_CSV.name} not found. Run get_best.py first.")
        return

    df = pd.read_csv(SUMMARY_CSV)
    print(f"Generating GIFs in '{mode}' mode...")

    for _, row in df.iterrows():
        metric = row["fitness_method"]
        name = str(row["pattern_name"]).lower()
        expr = int(row["experiment"])
        
        # Determine number of steps
        if mode == "best":
            num_steps = int(row["best_step"])
        else:
            num_steps = max_val

        # Locate the JSON ruleset
        sln_path = SOLUTIONS_ROOT / metric / "solns" / name / f"{name}_expr{expr}.json"

        if not sln_path.exists():
            print(f"  Warning: Solution not found: {sln_path}")
            continue

        with open(sln_path, "r") as f:
            data = json.load(f)

        # Prepare output path
        save_path = output_root / metric / f"{name}.gif"
        save_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"  Rendering {metric}/{name} for {num_steps} steps...")
        
        save_ca_gif(
            rules=data["rules"],
            steps=num_steps,
            grid_size=(17, 17),
            save_path=save_path,
            interval=125
        )

if __name__ == "__main__":
    main()