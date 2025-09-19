from typing import NamedTuple, Tuple, Dict, Any
import numpy as np

class GridPosition(NamedTuple):
    """3D grid position coordinates."""
    i: int  # x-coordinate
    j: int  # y-coordinate  
    k: int  # z-coordinate

class Displacement(NamedTuple):
    """2D horizontal displacement."""
    u: int  # x-displacement
    v: int  # y-displacement

class VerticalAction(NamedTuple):
    """Vertical control action."""
    action: int  # -1: down, 0: stay, +1: up

class GridConfig(NamedTuple):
    """Grid environment configuration."""
    n_x: int
    n_y: int
    n_z: int
    d_max: int  # Maximum displacement magnitude

    @classmethod
    def create(cls, n_x: int, n_y: int, n_z: int, d_max: int) -> 'GridConfig':
        """Create GridConfig with validation."""
        if n_x <= 0 or n_y <= 0 or n_z <= 0:
            raise ValueError("Grid dimensions must be positive integers")
        if d_max < 0:
            raise ValueError("Maximum displacement must be non-negative")
        if d_max >= min(n_x, n_y):
            raise ValueError("Maximum displacement should be smaller than grid dimensions")

        return cls(n_x, n_y, n_z, d_max)

# Type aliases
State = Dict[str, Any]
Observation = Dict[str, Any]
Info = Dict[str, Any]