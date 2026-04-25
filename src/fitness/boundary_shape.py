import numpy as np


def edge_mask(x):
    """
    Compute binary edge mask using 4-neighborhood differences.
    """
    e = np.zeros_like(x, dtype=bool)
    e[1:, :] |= x[1:, :] != x[:-1, :]
    e[:-1, :] |= x[:-1, :] != x[1:, :]
    e[:, 1:] |= x[:, 1:] != x[:, :-1]
    e[:, :-1] |= x[:, :-1] != x[:, 1:]
    return e


def boundary_iou(output, target):
    """
    Compute IoU between boundary masks.
    """
    o = edge_mask(output)
    t = edge_mask(target)

    intersection = np.logical_and(o, t).sum()
    union = np.logical_or(o, t).sum()

    return intersection / union if union else 0.0


def area_match(output, target):
    """
    Compare filled area between output and target.
    """
    ao = np.count_nonzero(output)
    at = np.count_nonzero(target)

    if at == 0:
        return 0.0

    return 1.0 - abs(ao - at) / at


def compute_fitness(grid, target, prev_grid=None):
    """
    Boundary-based structural fitness.

    This fitness prioritizes:
    - Matching shape boundaries
    - Matching overall filled area

    Characteristics:
    - Encourages structured patterns
    - Penalizes solid blobs
    - Emphasizes edges
    """

    boundary = boundary_iou(grid, target)
    area = area_match(grid, target)

    return 0.6 * boundary + 0.4 * area