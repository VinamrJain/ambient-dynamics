"""Type definitions for the grid environment."""

from typing import NamedTuple, Tuple, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np
import jax.numpy as jnp
from dataclasses import asdict

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
    
    Contains only truly universal fields that apply to ANY arena implementation.
    Using frozen dataclass for immutability and inheritance support.
    
    Fields:
        step_count: Number of steps taken in current episode
        last_action: Most recent action (None at episode start)
        last_reward: Reward from last step
        rng_key: JAX PRNG key for reproducibility
    """
    step_count: int
    last_action: Optional[int]
    last_reward: float
    rng_key: jnp.ndarray
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary (includes all fields from subclasses)."""
        return asdict(self)


@dataclass(frozen=True)
class GridArenaState(ArenaState):
    """Grid arena state (adds spatial state and grid-specific fields).
    
    Extends base state with grid world spatial information.
    
    Dynamic state:
        position: Current grid position
        last_position: Previous position (for trajectory tracking)
        last_displacement: Last observed field displacement
        out_of_bounds: Whether position violates boundaries
    
    Static config (for visualization/reproducibility):
        initial_position: Starting position for episode
    """
    position: GridPosition
    last_position: Optional[GridPosition]
    last_displacement: Optional[DisplacementObservation]
    out_of_bounds: bool
    initial_position: GridPosition  # Static: needed for visualization


@dataclass(frozen=True)
class NavigationArenaState(GridArenaState):
    """Navigation arena state (adds navigation task state and configuration).
    
    Extends grid state with navigation-specific information.
    
    Dynamic state:
        cumulative_reward: Total reward accumulated in episode
        target_reached: Whether target vicinity has been reached
    
    Static config (for visualization/analysis):
        target_position: Goal position
        vicinity_radius: Radius defining "reached" region
        distance_reward_weight: Weight for distance penalty (for analysis)
        vicinity_bonus: Reward for staying in vicinity (for analysis)
        step_penalty: Per-step penalty (for analysis)
        use_distance_decay: Whether vicinity bonus decays
        decay_rate: Exponential decay rate for vicinity bonus
    """
    # Dynamic navigation state
    cumulative_reward: float
    target_reached: bool
    
    # Static task configuration (for visualization/analysis)
    target_position: GridPosition
    vicinity_radius: float
    distance_reward_weight: float
    vicinity_bonus: float
    step_penalty: float
    use_distance_decay: bool
    decay_rate: float