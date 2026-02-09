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
    
    def __init__(self, config: GridConfig, d_max: int):
        """Initialize field with configuration.
        
        Args:
            config: Grid configuration specifying grid dimensions.
            d_max: Maximum displacement magnitude on ambient axes.
                   Displacements are clipped to [-d_max, d_max].
        """
        self.config = config
        self._d_max = d_max
        
        # Validate d_max against ambient dimensions
        if d_max < 0:
            raise ValueError("d_max must be non-negative")
        if self.ndim == 3:
            if d_max >= min(config.n_x, config.n_y):
                raise ValueError("d_max must be smaller than ambient dimensions")
        else:
            if d_max >= config.n_x:
                raise ValueError("d_max must be smaller than ambient dimension")
    
    @property
    def d_max(self) -> int:
        """Maximum displacement magnitude on ambient axes."""
        return self._d_max
    
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
    
    def get_mean_displacement_field(self) -> Optional[np.ndarray]:
        """Get underlying continuous field if available.
        
        Returns:
            Array with displacement values on ambient axes at each grid point
            - 3D: shape (n_x, n_y, n_z, 2) with (u, v) at each point
            - 2D: shape (n_x, n_y, 1) with (u,) at each point
        """
        return None

    def _clip_displacement(self, u: float, v: Optional[float] = None) -> DisplacementObservation:
        """Clip displacement values to [-d_max, d_max] bounds.
        
        Helper method for subclasses to ensure displacements stay within bounds.
        
        Args:
            u: Displacement on ambient axis 1.
            v: Displacement on ambient axis 2 (None for 2D).
            
        Returns:
            DisplacementObservation with clipped values.
        """
        u_clipped = float(max(-self._d_max, min(self._d_max, u)))
        if v is not None:
            v_clipped = float(max(-self._d_max, min(self._d_max, v)))
            return DisplacementObservation(u_clipped, v_clipped)
        return DisplacementObservation(u_clipped, None)