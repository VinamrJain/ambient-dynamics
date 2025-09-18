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

# Type aliases
State = Dict[str, Any]
Observation = Dict[str, Any]
Info = Dict[str, Any]