"""Abstract actor interface for vertical dynamics."""

from abc import ABC, abstractmethod
from typing import Optional
import numpy as np
import jax.numpy as jnp

from ..utils.types import GridPosition, GridConfig


class AbstractActor(ABC):
    """Abstract base class for actors with vertical dynamics.
    
    Actors control vertical (altitude) movement. Horizontal movement is
    determined by the field. No boundary enforcement - that's the arena's job.
    """
    
    def __init__(self, config: GridConfig):
        """Initialize actor with configuration.
        
        Args:
            config: Grid configuration.
        """
        self.config = config
    
    @abstractmethod
    def step_vertical(
        self, position: GridPosition, action: int, rng_key: jnp.ndarray
    ) -> GridPosition:
        """Apply vertical action to position (functional/stateless).
        
        Args:
            position: Current position.
            action: Vertical action (0=down, 1=stay, 2=up).
            rng_key: JAX PRNG key for stochastic dynamics.
            
        Returns:
            New position after vertical action (may be outside bounds).
        """
        pass
    
    @abstractmethod
    def get_vertical_displacement_pmf(self, action: int) -> np.ndarray:
        """Get PMF over vertical displacements for analysis (e.g., DP/VI).
        
        Args:
            action: Vertical action (0=down, 1=stay, 2=up).
            
        Returns:
            PMF array where entry [i] is P(z_displacement = i - z_max).
            For example, with actions in {-1, 0, +1}, this returns a 3-element array.
        """
        pass