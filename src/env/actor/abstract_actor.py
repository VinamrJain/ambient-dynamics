"""Abstract actor interface for vertical dynamics."""

from abc import ABC, abstractmethod
import numpy as np
import jax.numpy as jnp

from ..utils.types import GridPosition


class AbstractActor(ABC):
    """Abstract base class for actors with vertical dynamics.
    
    Actors control vertical (altitude) movement. Horizontal movement is
    determined by the field. Boundary enforcement is handled by the arena.
    """
    
    def __init__(self):
        """Initialize actor."""
        pass
    
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
    
    # Optional method for analysis
    def get_vertical_displacement_pmf(self) -> np.ndarray:
        """Get full PMF over vertical displacements for all actions.
        
        Returns:
            PMF array of shape (3, 2*z_max+1) where entry [a, j] is:
                P(z_displacement = j - z_max | action = a)
            
            where:
                - a ∈ {0, 1, 2} is action index (0=down, 1=stay, 2=up)
                - j ∈ {0, ..., 2*z_max} is displacement index
                - z_max is the maximum vertical displacement magnitude
            
            Example for z_max=2:
                - Shape: (3, 5) for displacements {-2, -1, 0, +1, +2}
                - pmf[0, 0] = P(z=-2 | action=0=down)
                - pmf[1, 2] = P(z=0 | action=1=stay)
                - pmf[2, 4] = P(z=+2 | action=2=up)
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support vertical displacement PMF analysis."
        )