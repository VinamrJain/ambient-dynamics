"""Type definitions for the grid environment."""

from typing import NamedTuple, Tuple, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np
import jax.numpy as jnp


class GridPosition(NamedTuple):
    """3D grid position coordinates (1-indexed)."""
    i: int  # x-coordinate [1, n_x]
    j: int  # y-coordinate [1, n_y]
    k: int  # z-coordinate (altitude) [1, n_z]


class DisplacementObservation(NamedTuple):
    """Observed displacement (can be continuous or discrete).
    
    Following BLE's units.py pattern: stores continuous values but provides
    integer properties for discrete state transitions.
    
    For continuous variant: u, v are floats from underlying field.
    For discrete variant: u, v are integers cast to float.
    """
    u: float  # x-displacement (continuous observation)
    v: float  # y-displacement (continuous observation)
    
    @property
    def u_int(self) -> int:
        """Get discrete x-displacement by rounding."""
        return int(round(self.u))
    
    @property
    def v_int(self) -> int:
        """Get discrete y-displacement by rounding."""
        return int(round(self.v))


class GridConfig(NamedTuple):
    """Grid environment configuration."""
    n_x: int  # Grid size in x dimension
    n_y: int  # Grid size in y dimension
    n_z: int  # Grid size in z dimension (altitude levels)
    d_max: int  # Maximum displacement magnitude in each direction

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


@dataclass(frozen=True)
class ArenaState:
    """Base arena state (common across all arena types).
    
    Using frozen dataclass for immutability and inheritance support.
    All arenas should track these core fields for reproducibility and analysis.
    """
    position: GridPosition
    last_position: Optional[GridPosition]
    step_count: int
    last_action: Optional[int]  # For logging/visualization
    last_reward: float  # For logging/visualization
    rng_key: jnp.ndarray  # For reproducibility/checkpointing
    out_of_bounds: bool = False


@dataclass(frozen=True)
class GridArenaState(ArenaState):
    """Grid arena state (adds field displacement observation).
    
    Extends base state with grid-specific displacement observations.
    """
    last_displacement: Optional[DisplacementObservation] = None


@dataclass(frozen=True)
class NavigationArenaState(GridArenaState):
    """Navigation arena state (adds navigation-specific fields).
    
    Extends grid state with navigation task information.
    """
    target_position: GridPosition = None
    vicinity_radius: float = 0.0
    cumulative_reward: float = 0.0
    target_reached: bool = False