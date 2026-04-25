import json
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to path for imports
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.ca import CellularAutomata
from src.utils import PROJECT_ROOT, create_seed

SUMMARY_CSV = Path(__file__).parent / "best_solns_summary.csv"
OUTPUT_DIR = Path(__file__).parent / "best_frames"
GRID_SIZE = (17, 17)

def save_freeze_frame(grid, path):
    h, w = grid.shape
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.imshow(grid, cmap="binary", interpolation='nearest')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticks(np.arange(-.5, w, 1), minor=True)
    ax.set_yticks(np.arange(-.5, h, 1), minor=True)
    ax.grid(which="minor", color="black", linestyle='-', linewidth=0.5, alpha=0.25)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

def main():
    if not SUMMARY_CSV.exists():
        print(f"Error: Run get_best.py first to generate {SUMMARY_CSV}")
        return

    df = pd.read_csv(SUMMARY_CSV)
    print(f"Generating {len(df)} freeze frames...")

    for _, row in df.iterrows():
        metric = row["fitness_method"]
        name = str(row["pattern_name"]).lower()
        
        # Locate JSON ruleset
        json_path = (
            PROJECT_ROOT / "outputs" / metric / "solns" 
            / name / f"{name}_expr{row['experiment']}.json"
        )

        if not json_path.exists():
            print(f"  Warning: Ruleset not found at {json_path}")
            continue

        # Run CA
        with open(json_path) as f:
            data = json.load(f)
        
        ca = CellularAutomata(data["rules"])
        seed = create_seed(GRID_SIZE)
        final_grid = ca.run(seed, int(row["best_step"]))

        # Save
        out_path = OUTPUT_DIR / metric / f"{name}.png"
        save_freeze_frame(final_grid, out_path)
        print(f"  Saved: {metric}/{name}.png")

if __name__ == "__main__":
    main()