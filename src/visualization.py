import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap
import numpy as np
from pathlib import Path
from src.ca import CellularAutomata
from src.utils import create_seed

def display_grid(grid: np.ndarray, colors: dict):
    sorted_keys = sorted(colors.keys())
    colors_list = [colors[i] for i in sorted_keys]
    color_map = ListedColormap(colors_list)

    fig, ax = plt.subplots()
    ax.imshow(grid, cmap=color_map, interpolation='nearest')
    ax.set_axis_off()
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.show()

def save_grid(grid: np.ndarray, colors: dict, path):
    sorted_keys = sorted(colors.keys())
    colors_list = [colors[i] for i in sorted_keys]
    color_map = ListedColormap(colors_list)

    h, w = grid.shape

    fig, ax = plt.subplots(figsize=(3,3))

    ax.imshow(grid, cmap=color_map, interpolation='nearest')

    # Remove ticks
    ax.set_xticks([])
    ax.set_yticks([])

    # Set grid lines
    ax.set_xticks(np.arange(-.5, w, 1), minor=True)
    ax.set_yticks(np.arange(-.5, h, 1), minor=True)

    ax.grid(
        which="minor",
        color="black",
        linestyle='-',
        linewidth=0.5,
        alpha=0.25
    )

    # Remove padding completely
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    plt.savefig(
        path,
        dpi=300,
        bbox_inches='tight',
        pad_inches=0
    )

    plt.close(fig)

def save_ca_gif(rules: list, steps: int, grid_size: tuple, save_path: Path, interval: int = 100, pause_frames: int = 15):
    """
    Runs a CA simulation and saves the evolution as a GIF.
    """
    
    ca = CellularAutomata(rules)
    grid = create_seed(grid_size)
    
    # Get the actual simulation frames
    frames = ca.run(grid, steps, return_frames=True)

    # Add stall by uplicate the last frame
    if len(frames) > 0:
        last_frame = frames[-1]
        for _ in range(pause_frames):
            frames.append(last_frame)

    h, w = frames[0].shape
    fig, ax = plt.subplots(figsize=(4,4))

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")

    ax.set_xticks(np.arange(-.5, w, 1), minor=True)
    ax.set_yticks(np.arange(-.5, h, 1), minor=True)
    ax.grid(
        which="minor",
        color="black",
        linestyle='-',
        linewidth=0.4,
        alpha=0.25
    )

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    im = ax.imshow(
        frames[0],
        cmap="binary",
        interpolation="nearest",
        animated=True
    )

    def update(frame):
        im.set_array(frame)
        return [im]

    ani = FuncAnimation(
        fig,
        update,
        frames=frames,
        interval=interval,
        blit=True
    )

    save_path.parent.mkdir(parents=True, exist_ok=True)

    ani.save(
        save_path,
        writer="pillow",
        fps=1000 // interval,
        dpi=100,
        savefig_kwargs={
            "bbox_inches": "tight",
            "pad_inches": 0
        }
    )

    plt.close(fig)