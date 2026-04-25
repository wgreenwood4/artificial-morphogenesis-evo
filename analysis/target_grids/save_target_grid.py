import sys
from pathlib import Path
import matplotlib.pyplot as plt

root_path = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(root_path))

from src.pattern import Pattern
from src.visualization import save_grid
from src.utils import INPUTS_DIR

def process_file(file_path):
    if not file_path.exists():
        print(f"Error: Could not find file at {file_path}")
        return

    print(f"Processing: {file_path.name}...")
    p = Pattern.from_file(file_path)
    
    output_name = f"{file_path.stem}.png"
    save_grid(p.grid, p.colors, output_name)
    print(f"Saved to: {output_name}")

if len(sys.argv) < 2:
    print("Usage: python save_target_grid.py [filename.txt | all]")
    sys.exit(1)

user_input = sys.argv[1].lower()
plt.style.use("seaborn-v0_8-whitegrid")

if user_input == "all":
    files_to_process = list(INPUTS_DIR.glob("*.txt"))
    if not files_to_process:
        print(f"No .txt files found in {INPUTS_DIR}")
    else:
        for f in files_to_process:
            process_file(f)
else:
    if not user_input.endswith(".txt"):
        user_input += ".txt"
    
    target_path = INPUTS_DIR / user_input
    process_file(target_path)