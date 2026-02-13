"""Grid actor with clipped-Gaussian controllable axis dynamics.
"""

import numpy as np
import jax
import jax.numpy as jnp
from scipy.stats import norm

from .abstract_actor import AbstractActor
from ..utils.types import GridPosition


class GridActor(AbstractActor):
    """Grid actor with clipped-Gaussian controllable axis dynamics.
    
    Supports both 2D and 3D settings:
    - 3D: Controls z-axis (k coordinate)
    - 2D: Controls y-axis (j coordinate)
    
    Dynamics:
        Given action a in {0, 1, 2}, intended displacement = scale * (a - 1).
        Continuous signal: w_tilde = clip(scale*(a-1) + eps, -z_max, z_max), eps ~ N(0, noise_std^2)
        Discrete displacement: Delta = round(w_tilde)
    """
    
    def __init__(
        self,
        scale: float = 1.0,
        noise_std: float = 0.1,
        z_max: int = 1
    ):
        """Initialize grid actor.
        
        Args:
            scale: Multiplier for the intended displacement.
                   Intended displacement = scale * (action - 1).
            noise_std: Standard deviation of additive Gaussian noise.
            z_max: Maximum controllable displacement magnitude.
                   Displacements are clipped to [-z_max, z_max].
        """
        super().__init__()
        
        if scale <= 0.0:
            raise ValueError(f"scale must be positive, got {scale}")
        if noise_std < 0.0:
            raise ValueError(f"noise_std must be non-negative, got {noise_std}")
        if z_max < 0:
            raise ValueError(f"z_max must be non-negative, got {z_max}")
        
        self.scale = float(scale)
        self.noise_std = float(noise_std)
        self.z_max = int(z_max)
    
    def step_controllable(
        self, position: GridPosition, action: int, rng_key: jnp.ndarray
    ) -> GridPosition:
        """Apply action on controllable axis with clipped-Gaussian noise.
        
        Sampling: w_tilde = clip(scale*(action-1) + eps, -z_max, z_max)  eps ~ N(0, noise_std^2)
                  Delta = round(w_tilde)
        
        Args:
            position: Current position.
            action: Controllable axis action (0=decrease, 1=stay, 2=increase).
            rng_key: JAX PRNG key for stochastic dynamics.
            
        Returns:
            New position after action (may be outside grid bounds).
        """
        # Mean displacement for this action
        mean = self.scale * (action - 1)
        
        # Sample noise and compute continuous signal
        noise = float(jax.random.normal(rng_key) * self.noise_std)
        w_continuous = mean + noise
        
        # Clip to [-z_max, z_max]
        w_clipped = max(-self.z_max, min(self.z_max, w_continuous))
        
        # Round to integer displacement
        displacement = int(round(w_clipped))
        
        # Apply displacement to controllable axis
        new_controllable = position.controllable + displacement
        
        # Return new position based on dimensionality
        if position.ndim == 3:
            return GridPosition(position.i, position.j, new_controllable)
        else:
            return GridPosition(position.i, new_controllable, None)
    
    def get_controllable_displacement_pmf(self) -> np.ndarray:
        """Get PMF over controllable displacements for all actions.
        
        Uses the same clipped-Gaussian PMF as the field:
            P(Delta = k | action = a) = clipped_gaussian_pmf(k; m_a, sigma_a, z_max)
        where m_a = scale * (a - 1).
        
        Returns:
            PMF array of shape (3, 2*z_max+1) where entry [a, j] is
            P(displacement = j - z_max | action = a).
            
            Displacements range over {-z_max, ..., +z_max}.
        """
        n_actions = 3
        n_displacements = 2 * self.z_max + 1
        pmf = np.zeros((n_actions, n_displacements), dtype=np.float32)
        
        for action in range(n_actions):
            mean = self.scale * (action - 1)
            pmf[action, :] = self._clipped_gaussian_pmf(mean)
        
        return pmf
    
    def _clipped_gaussian_pmf(self, mean: float) -> np.ndarray:
        """Compute 1D clipped-Gaussian PMF.
        
        Matches the field's PMF derivation:
        - Boundary bins accumulate tail probability beyond ±z_max.
        - Interior bins integrate the Gaussian over [k-0.5, k+0.5].
        
        Args:
            mean: Mean of the Gaussian (m_a = scale * (action - 1)).
            
        Returns:
            PMF array of shape (2*z_max+1,).
        """
        z = self.z_max
        sigma = self.noise_std
        n = 2 * z + 1
        
        # Degenerate case: z_max = 0
        if z == 0:
            return np.ones(1, dtype=np.float32)
        
        # Deterministic case: sigma_a = 0
        if sigma == 0.0:
            clipped = max(-z, min(z, mean))
            rounded = int(round(clipped))
            pmf = np.zeros(n, dtype=np.float32)
            pmf[rounded + z] = 1.0
            return pmf
        
        # General case: clipped Gaussian
        pmf = np.zeros(n, dtype=np.float32)
        
        # Left boundary: P(continuous < -z_max + 0.5)
        pmf[0] = norm.cdf((-z + 0.5 - mean) / sigma)
        
        # Interior bins: P(k - 0.5 <= continuous < k + 0.5)
        if z > 0:
            interior_k = np.arange(-z + 1, z)
            upper = (interior_k + 0.5 - mean) / sigma
            lower = (interior_k - 0.5 - mean) / sigma
            pmf[1:-1] = norm.cdf(upper) - norm.cdf(lower)
        
        # Right boundary: P(continuous >= z_max - 0.5)
        pmf[-1] = 1.0 - norm.cdf((z - 0.5 - mean) / sigma)
        
        return pmf
