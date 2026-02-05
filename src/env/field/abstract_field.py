"""Abstract field interface for environmental dynamics on ambient axes."""

from abc import ABC, abstractmethod
from typing import Optional, Tuple
import numpy as np
import jax.numpy as jnp

from ..utils.types import GridPosition, DisplacementObservation, GridConfig


class AbstractField(ABC):
    """Abstract base class for environmental fields.
    
    Fields represent environmental forces that produce displacements on ambient axes.
    Supports both 2D (1 ambient axis) and 3D (2 ambient axes) settings.
    Can support both discrete and continuous displacement observations.
    """
    
    def __init__(self, config: GridConfig):
        """Initialize field with configuration.
        
        Args:
            config: Grid configuration specifying dimensions and displacement bounds.
        """
        self.config = config
    
    @property
    def ndim(self) -> int:
        """Number of spatial dimensions (2 or 3)."""
        return self.config.ndim
    
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
        """Sample displacement on ambient axes at given position.
        
        Args:
            position: Current grid position.
            rng_key: JAX PRNG key for sampling.
            
        Returns:
            Displacement observation:
            - 3D: (u, v) displacement on ambient axes
            - 2D: (u, None) displacement on single ambient axis
        """
        pass
    
    # Optional methods for analysis (not required for all fields)
    
    def get_displacement_pmf(self, position: GridPosition) -> Optional[np.ndarray]:
        """Get displacement PMF at position
        
        Args:
            position: Grid position to query.
            
        Returns:
            PMF array or None if not available.
            - 3D: shape (2*d_max+1, 2*d_max+1), entry [i,j] = P(u=i-d_max, v=j-d_max)
            - 2D: shape (2*d_max+1,), entry [i] = P(u=i-d_max)
        """
        return None
    
    def get_mean_displacement(self, position: GridPosition) -> Optional[Tuple[float, ...]]:
        """Get expected displacement at a position
        
        Args:
            position: Grid position to query.
            
        Returns:
            Mean displacement tuple or None if not available.
            - 3D: (u_mean, v_mean) on ambient axes
            - 2D: (u_mean,) on single ambient axis
        """
        return None
    
    def get_continuous_field(self) -> Optional[np.ndarray]:
        """Get underlying continuous field if available.
        
        Returns:
            Array with displacement values on ambient axes at each grid point
            - 3D: shape (n_x, n_y, n_z, 2) with (u, v) at each point
            - 2D: shape (n_x, n_y, 1) with (u,) at each point
        """
        return None