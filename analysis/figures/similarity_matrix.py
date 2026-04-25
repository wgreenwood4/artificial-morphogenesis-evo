import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

from src.pattern import Pattern
from src.fitness.pixel_iou import binary_iou
from src.utils import INPUTS_DIR, PROJECT_ROOT

# Setup Directories
FIG_DIR = PROJECT_ROOT / "analysis" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Automatically get all patterns in the inputs folder
input_files = sorted(list(INPUTS_DIR.glob("*.txt")))
patterns = [f.stem for f in input_files]
labels = [p.replace("_", " ").title() for p in patterns]

# Load all patterns into a dictionary
pattern_grids = {}
for f in input_files:
    p = Pattern.from_file(f)
    pattern_grids[f.stem] = p.grid
    print(f"Loaded {f.stem}: {p.grid.shape}")

# Compute IoU matrix
n = len(patterns)
iou_matrix = np.zeros((n, n))

for i, p1 in enumerate(patterns):
    for j, p2 in enumerate(patterns):
        iou_matrix[i, j] = binary_iou(pattern_grids[p1], pattern_grids[p2])

plot_data = iou_matrix.copy()
np.fill_diagonal(plot_data, np.nan)

custom_cmap = LinearSegmentedColormap.from_list("GYR", ["#228B22", "#FFFF00", "#FF0000"]).copy()
custom_cmap.set_bad(color='#EEEEEE') # Gray for the diagonal

fig, ax = plt.subplots(figsize=(max(6, n*0.8), max(6, n*0.8)))
im = ax.imshow(plot_data, vmin=0, vmax=1, cmap=custom_cmap)

# Add values inside cells
for i in range(n):
    for j in range(n):
        if i == j:
            ax.text(j, i, "--", ha="center", va="center", color="#888888")
        else:
            val = iou_matrix[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color="black")

# Formatting
ax.set_xticks(np.arange(n))
ax.set_yticks(np.arange(n))
ax.set_xticklabels(labels, rotation=45, ha="right")
ax.set_yticklabels(labels)
ax.grid(False)

# Colorbar
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Similarity Score (IoU)")

plt.tight_layout()
output_path = FIG_DIR / "pattern_iou_matrix.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")
print(f"Success! Matrix saved to {output_path}")
plt.close(fig)