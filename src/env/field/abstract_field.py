"""Abstract field interface for environmental dynamics."""

from abc import ABC, abstractmethod
from typing import Optional
import numpy as np
import jax.numpy as jnp

from ..utils.types import GridPosition, DisplacementObservation, GridConfig


class AbstractField(ABC):
    """Abstract base class for environmental fields.
    
    Fields represent environmental forces that produce horizontal displacements.
    Can support both discrete and continuous displacement observations.
    """
    
    def __init__(self, config: GridConfig):
        """Initialize field with configuration.
        
        Args:
            config: Grid configuration specifying dimensions and displacement bounds.
        """
        self.config = config
    
    @abstractmethod
    def reset(self, rng_key: jnp.ndarray) -> None:
        """Reset/regenerate the field configuration.
        
        Args:
            rng_key: JAX PRNG key for reproducible randomness.
        """
        pass
    
    @abstractmethod
    def sample_displacement(
        self, position: GridPosition, rng_key: jnp.ndarray
    ) -> DisplacementObservation:
        """Sample horizontal displacement at given position.
        
        Args:
            position: Current grid position.
            rng_key: JAX PRNG key for sampling.
            
        Returns:
            Displacement observation (can be continuous or discrete).
        """
        pass
    
    # Optional methods for analysis (not required for all fields)
    
    def get_displacement_pmf(self, position: GridPosition) -> Optional[np.ndarray]:
        """Get displacement PMF at position for analysis (e.g., DP/VI).
        
        Args:
            position: Grid position to query.
            
        Returns:
            PMF array of shape (2*d_max+1, 2*d_max+1) or None if not available.
            Entry [i, j] corresponds to P(u=i-d_max, v=j-d_max).
        """
        return None
    
    def get_continuous_field(self) -> Optional[np.ndarray]:
        """Get underlying continuous field if available.
        
        Returns:
            Array of shape (n_x, n_y, n_z, 2) with (u, v) at each point,
            or None if field is purely discrete.
        """
        return None