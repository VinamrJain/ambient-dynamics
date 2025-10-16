"""Simple field implementation with uniform random displacements."""

import numpy as np
import jax
import jax.numpy as jnp
from typing import Optional

from .abstract_field import AbstractField
from ..utils.types import GridPosition, DisplacementObservation, GridConfig


class SimpleField(AbstractField):
    """Simple field with uniform random displacements (discrete).
    
    Each position experiences independent uniform random displacement
    in {-d_max, ..., +d_max} x {-d_max, ..., +d_max}.
    """
    
    def __init__(self, config: GridConfig):
        """Initialize simple field.
        
        Args:
            config: Grid configuration.
        """
        super().__init__(config)
        self._rng_key = None
    
    def reset(self, rng_key: jnp.ndarray) -> None:
        """Reset field with new RNG key."""
        self._rng_key = rng_key
    
    def sample_displacement(
        self, position: GridPosition, rng_key: jnp.ndarray
    ) -> DisplacementObservation:
        """Sample uniform random displacement (discrete).
        
        Args:
            position: Current grid position (unused in this simple field).
            rng_key: JAX PRNG key for sampling.
            
        Returns:
            Displacement observation with discrete integer values.
        """
        # Split key for u and v sampling
        key_u, key_v = jax.random.split(rng_key)
        
        # Sample uniformly from {-d_max, ..., +d_max}
        u = jax.random.randint(
            key_u, shape=(), 
            minval=-self.config.d_max, 
            maxval=self.config.d_max + 1
        )
        v = jax.random.randint(
            key_v, shape=(), 
            minval=-self.config.d_max, 
            maxval=self.config.d_max + 1
        )
        
        # Convert to float for DisplacementObservation
        return DisplacementObservation(float(u), float(v))
    
    def get_displacement_pmf(self, position: GridPosition) -> np.ndarray:
        """Return uniform PMF over displacement space.
        
        Returns:
            Array of shape (2*d_max+1, 2*d_max+1) with uniform probabilities.
        """
        size = 2 * self.config.d_max + 1
        return np.ones((size, size), dtype=np.float32) / (size ** 2)