import numpy as np
from scipy.ndimage import distance_transform_edt


def distance_shape_match(output, target):
    """
    Compare shapes using distance transform similarity.
    """
    output_bin = output > 0
    target_bin = target > 0

    target_dist = distance_transform_edt(~target_bin)
    output_dist = distance_transform_edt(~output_bin)

    diff = np.abs(target_dist - output_dist)

    return max(0.0, 1.0 - (np.mean(diff) / max(target.shape)))


def downsample(grid, factor):
    """
    Downsample grid for multiscale comparison.
    """
    h, w = grid.shape

    h_trim = (h // factor) * factor
    w_trim = (w // factor) * factor

    grid = grid[:h_trim, :w_trim]

    return grid.reshape(
        h_trim // factor, factor,
        w_trim // factor, factor
    ).mean(axis=(1, 3)) > 0.5


def multiscale_shape_match(output, target):
    """
    Compare shapes at multiple spatial scales.
    """
    scales = [1, 2, 4]
    scores = []

    for s in scales:
        if s == 1:
            o = output
            t = target
        else:
            if output.shape[0] % s != 0:
                continue
            o = downsample(output, s)
            t = downsample(target, s)

        scores.append(distance_shape_match(o, t))

    return float(np.mean(scores)) if scores else 0.0


def compute_fitness(grid, target, prev_grid=None):
    """
    Distance-transform based morphological fitness.

    This fitness evaluates global shape similarity using
    distance transforms at multiple spatial scales.

    Characteristics:
    - Encourages global morphology
    - Allows flexible internal structure
    - Robust to small pixel differences
    """

    return multiscale_shape_match(grid, target)