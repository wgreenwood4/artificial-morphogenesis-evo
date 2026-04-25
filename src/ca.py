import numpy as np

class CellularAutomata:
    def __init__(self, rules=None):
        if rules is None:
            self.rules = np.random.randint(0, 2, size=512, dtype=np.uint8)
            self.rules[0] = 0
        else:
            self.rules = np.asarray(rules, dtype=np.uint8).copy()

    def step(self, grid):
        grid = np.asarray(grid, dtype=np.uint8)

        neighbors = np.stack([
            np.roll(np.roll(grid, i, axis=0), j, axis=1)
            for i in (-1, 0, 1)
            for j in (-1, 0, 1)
        ], axis=-1)

        codes = np.zeros(grid.shape, dtype=np.uint16)
        for k in range(9):
            codes = (codes << 1) | neighbors[..., k].astype(np.uint16, copy=False)

        return self.rules[codes]

    def run(self, grid, steps, return_frames=False):
        grid = np.asarray(grid, dtype=np.uint8)

        if return_frames:
            frames = [grid.copy()]

        for _ in range(steps):
            grid = self.step(grid)
            if return_frames:
                frames.append(grid.copy())

        return frames if return_frames else grid