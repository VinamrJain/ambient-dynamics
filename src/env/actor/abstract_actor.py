"""Abstract actor interface for controllable axis dynamics."""

from abc import ABC, abstractmethod
import numpy as np
import jax.numpy as jnp

from ..utils.types import GridPosition


class AbstractActor(ABC):
    """Abstract base class for actors with controllable axis dynamics.
    
    Actors control movement on the controllable axis (z for 3D, y for 2D).
    Movement on ambient axes is determined by the field.
    Boundary enforcement is handled by the arena.
    """
    
    def __init__(self):
        """Initialize actor."""
        pass
    
    @abstractmethod
    def step_controllable(
        self, position: GridPosition, action: int, rng_key: jnp.ndarray
    ) -> GridPosition:
        """Apply action on controllable axis (functional/stateless).
        
        Args:
            position: Current position.
            action: Controllable axis action (0=decrease, 1=stay, 2=increase).
            rng_key: JAX PRNG key for stochastic dynamics.
            
        Returns:
            New position after action (may be outside bounds).
        """
        pass
    
    # Optional method for analysis
    def get_controllable_displacement_pmf(self) -> np.ndarray:
        """Get full PMF over controllable axis displacements for all actions.
        
        Returns:
            PMF array of shape (3, 2*z_max+1) where entry [a, j] is:
                P(controllable_displacement = j - z_max | action = a)
            
            where:
                - a ∈ {0, 1, 2} is action index (0=decrease, 1=stay, 2=increase)
                - j ∈ {0, ..., 2*z_max} is displacement index
                - z_max is the actor's maximum controllable displacement magnitude
            
            Example for z_max=2 (displacements in {-2, -1, 0, +1, +2}):
                pmf has shape (3, 5) where columns index d + z_max
                
                pmf[0, 0] = P(d=-2 | action=0)  # decrease action, max negative displacement
                pmf[1, 2] = P(d=0 | action=1)   # stay action, zero displacement
                pmf[2, 4] = P(d=+2 | action=2)  # increase action, max positive displacement
                
                Each row sums to 1.0.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support controllable displacement PMF analysis."
        )