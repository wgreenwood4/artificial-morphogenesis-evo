import numpy as np


def binary_iou(output, target):
    """
    Compute binary Intersection-over-Union (IoU).

    Measures pixel-level similarity between output and target grids.
    """
    output_bin = output.astype(bool)
    target_bin = target.astype(bool)

    intersection = np.logical_and(output_bin, target_bin).sum()
    union = np.logical_or(output_bin, target_bin).sum()

    return intersection / union if union else 0.0


def compute_fitness(grid, target, prev_grid=None):
    """
    Pixel-based fitness using binary IoU.

    This fitness evaluates only pixel-level similarity between the
    evolved pattern and the target pattern.

    Characteristics:
    - Encourages filled regions
    - Favors simple shapes
    - Fast convergence
    """

    return binary_iou(grid, target)