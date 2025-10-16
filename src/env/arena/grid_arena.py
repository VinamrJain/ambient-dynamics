"""Basic grid arena implementation with boundary handling."""

from typing import Tuple
import gymnasium as gym
import numpy as np
import jax
import jax.numpy as jnp

from .abstract_arena import AbstractArena
from ..field.abstract_field import AbstractField
from ..actor.abstract_actor import AbstractActor
from ..utils.types import (
    GridPosition, DisplacementObservation, GridConfig, ArenaState
)


class GridArena(AbstractArena):
    """Basic grid arena with configurable boundary handling.
    
    Simulates an actor moving in a 3D grid under environmental field forces.
    Supports different boundary conditions and serves as base for specific tasks.
    """
    
    def __init__(
        self,
        field: AbstractField,
        actor: AbstractActor,
        config: GridConfig,
        initial_position: GridPosition,
        boundary_mode: str = 'clip'
    ):
        """Initialize grid arena.
        
        Args:
            field: Environmental field providing displacements.
            actor: Actor with vertical dynamics.
            config: Grid configuration.
            initial_position: Starting position for reset.
            boundary_mode: How to handle boundaries:
                - 'clip': Clamp position to valid range (default)
                - 'periodic': Wrap around (toroidal topology)
                - 'terminal': Mark as terminal when crossing boundary
        """
        self.field = field
        self.actor = actor
        self.config = config
        self.initial_position = initial_position
        self.boundary_mode = boundary_mode
        
        # Validate boundary mode
        valid_modes = ['clip', 'periodic', 'terminal']
        if boundary_mode not in valid_modes:
            raise ValueError(
                f"boundary_mode must be one of {valid_modes}, got {boundary_mode}"
            )
        
        # Arena state (updated in reset and step)
        self.position = initial_position
        self.last_position = initial_position
        self.last_displacement = DisplacementObservation(0.0, 0.0)
        self.step_count = 0
        self._out_of_bounds = False
        self._rng = None
    
    def reset(self, rng_key: jnp.ndarray) -> np.ndarray:
        """Reset arena to initial state."""
        # Split RNG for field and future use
        self._rng = rng_key
        field_key, self._rng = jax.random.split(self._rng)
        
        # Reset field
        self.field.reset(field_key)
        
        # Reset state
        self.position = self.initial_position
        self.last_position = self.initial_position
        self.last_displacement = DisplacementObservation(0.0, 0.0)
        self.step_count = 0
        self._out_of_bounds = False
        
        return self._get_observation()
    
    def step(self, action: int) -> np.ndarray:
        """Execute one simulation step."""
        # Split RNG keys for field and actor
        field_key, actor_key, self._rng = jax.random.split(self._rng, 3)
        
        # 1. Sample horizontal displacement from field (continuous observation)
        displacement_obs = self.field.sample_displacement(
            self.position, field_key
        )
        
        # 2. Apply horizontal displacement (discrete state transition)
        du, dv = displacement_obs.to_discrete()
        new_i = self.position.i + du
        new_j = self.position.j + dv
        new_k = self.position.k
        
        # Store last position before update
        self.last_position = self.position
        
        # Update horizontal position (before boundary check)
        self.position = GridPosition(new_i, new_j, new_k)
        
        # 3. Apply vertical action
        self.position = self.actor.step_vertical(self.position, action, actor_key)
        
        # 4. Enforce boundaries
        self.position, self._out_of_bounds = self._enforce_boundaries(
            self.position
        )
        
        # Store displacement observation for next observation
        self.last_displacement = displacement_obs
        
        self.step_count += 1
        
        return self._get_observation()
    
    def get_state(self) -> ArenaState:
        """Get complete arena state."""
        return ArenaState(
            position=self.position,
            last_position=self.last_position,
            last_displacement=self.last_displacement,
            step_count=self.step_count,
            out_of_bounds=self._out_of_bounds
        )
    
    def set_state(self, state: ArenaState) -> None:
        """Restore arena state."""
        self.position = state.position
        self.last_position = state.last_position
        self.last_displacement = state.last_displacement
        self.step_count = state.step_count
        self._out_of_bounds = state.out_of_bounds
    
    def compute_reward(self) -> float:
        """Default reward (override in subclasses for specific tasks)."""
        return 0.0
    
    def is_terminal(self) -> bool:
        """Default termination logic (override in subclasses)."""
        if self.boundary_mode == 'terminal':
            return self._out_of_bounds
        return False
    
    @property
    def observation_space(self) -> gym.Space:
        """Observation space: [i, j, k, u_obs, v_obs]."""
        return gym.spaces.Box(
            low=np.array([
                1, 1, 1,
                -self.config.d_max, -self.config.d_max
            ], dtype=np.float32),
            high=np.array([
                self.config.n_x, self.config.n_y, self.config.n_z,
                self.config.d_max, self.config.d_max
            ], dtype=np.float32),
            dtype=np.float32
        )
    
    def _get_observation(self) -> np.ndarray:
        """Construct flat observation array."""
        return np.array([
            float(self.position.i),
            float(self.position.j),
            float(self.position.k),
            self.last_displacement.u,
            self.last_displacement.v
        ], dtype=np.float32)
    
    def _enforce_boundaries(
        self, position: GridPosition
    ) -> Tuple[GridPosition, bool]:
        """Enforce boundary conditions based on mode.
        
        Returns:
            (new_position, out_of_bounds_flag)
        """
        out_of_bounds = False
        
        if self.boundary_mode == 'clip':
            # Clamp to valid range
            new_i = int(max(1, min(position.i, self.config.n_x)))
            new_j = int(max(1, min(position.j, self.config.n_y)))
            new_k = int(max(1, min(position.k, self.config.n_z)))
            
            # Check if clamping occurred
            out_of_bounds = (
                new_i != position.i or 
                new_j != position.j or 
                new_k != position.k
            )
            position = GridPosition(new_i, new_j, new_k)
            
        elif self.boundary_mode == 'periodic':
            # Wrap around (toroidal in x-y, clip in z)
            new_i = ((position.i - 1) % self.config.n_x) + 1
            new_j = ((position.j - 1) % self.config.n_y) + 1
            new_k = int(max(1, min(position.k, self.config.n_z)))
            position = GridPosition(new_i, new_j, new_k)
            
        elif self.boundary_mode == 'terminal':
            # Check if outside bounds
            out_of_bounds = (
                position.i < 1 or position.i > self.config.n_x or
                position.j < 1 or position.j > self.config.n_y or
                position.k < 1 or position.k > self.config.n_z
            )
            # Don't modify position - arena will terminate
        
        return position, out_of_bounds

