import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import ast

from src.utils import PROJECT_ROOT

def aggregate_best_fitness(df):
    return df.groupby(["name", "generation"])["best_in_gen"].agg(["mean", "std"]).reset_index()

def aggregate_steps(df):
    df = df.copy()
    df["avg_steps"] = df["gen_steps"].apply(lambda x: np.mean(ast.literal_eval(x)))
    return df.groupby(["name", "generation"])["avg_steps"].agg(["mean", "std"]).reset_index()

def plot_metric_subplot(ax, agg_df, title, ylabel, ylim, shade):
    patterns = sorted(agg_df["name"].unique())
    for pattern in patterns:
        subset = agg_df[agg_df["name"] == pattern]
        x, y, std = subset["generation"], subset["mean"], subset["std"]
        ax.plot(x, y, label=pattern)
        if shade:
            ax.fill_between(x, y - std, y + std, alpha=0.2)
    
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Generation")
    ax.set_ylim(ylim)

def save_figure(dfs, agg_func, ylabel, ylim, filename, shade=True):
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharex=True)
    
    for ax, (metric_name, df) in zip(axes, dfs.items()):
        agg = agg_func(df)
        plot_metric_subplot(ax, agg, metric_name, ylabel, ylim, shade)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right", bbox_to_anchor=(0.98, 0.5), fontsize=10)
    
    plt.tight_layout(rect=[0, 0, 0.88, 0.95])
    save_path = FIG_DIR / filename
    plt.savefig(save_path, dpi=300)
    print(f"Saved figure: {save_path}")
    plt.close(fig)


plt.style.use("seaborn-v0_8-whitegrid")
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIG_DIR = PROJECT_ROOT / "analysis" / "figures"
FIG_DIR.mkdir(exist_ok=True, parents=True)

# Load Data
fitness_files = {
    "Pixel IoU": OUTPUT_DIR / "pixel_iou/pixel_iou.csv",
    "Boundary IoU": OUTPUT_DIR / "boundary_shape/boundary_shape.csv",
    "Distance Transformation": OUTPUT_DIR / "distance_morphology/distance_morphology.csv",
}
loaded_dfs = {name: pd.read_csv(path) for name, path in fitness_files.items() if path.exists()}

# Generate Figures
if loaded_dfs:
    # Best Fitness Figure
    save_figure(
        loaded_dfs, aggregate_best_fitness, 
        "Best Fitness", [0.3, 1.0], "best_fitness.png"
    )

    # CA Steps Figure
    save_figure(
        loaded_dfs, aggregate_steps, 
        "Avg CA Steps", [0.0, 40.0], "avg_ca_steps.png", shade=False
    )
else:
    print("No data files found to plot.")