from typing import Optional, Dict
import numpy as np

from .utils import load_pattern

class Pattern:
    def __init__(
        self, 
        grid: np.ndarray, 
        name: Optional[str] = None, 
        shape: Optional[str] = None, 
        colors: Optional[Dict[int, str]] = None
    ):
        self.grid = grid
        self.name = name
        self.shape = shape
        self.colors = colors
    
    @classmethod
    def from_file(cls, path: str):
        metadata, grid = load_pattern(path)
        return cls(grid=grid, **metadata)
    
    @property
    def size(self):
        return self.grid.shape

    @property
    def n_colors(self):
        return len(set(self.grid.flatten()))
